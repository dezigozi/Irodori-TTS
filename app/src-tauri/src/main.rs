// いろとりスタジオ — Irodori-TTS の常駐サーバ(:3952)を使う声のデータベース＆読み上げ保存アプリ
// サーバとの会話（話者追加・合成・レンダー）はフロントの fetch が直接やる。
// Rust 側は「サーバの起動」「設定の保存」「ファイル/Finder まわり」だけを受け持つ。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Manager};

const SERVER_URL: &str = "http://127.0.0.1:3952";
const IRODORI_REPO: &str = "git/Irodori-TTS";
const DEFAULT_OUT_SUBDIR: &str = "Music/いろとりスタジオ";

#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct Settings {
    #[serde(default)]
    output_dir: String,
    #[serde(default)]
    speaker: String,
    #[serde(default)]
    expression: String,
    #[serde(default)]
    rate: Option<f64>,
    // 原稿AI（⌘の 🤖 ボタン）の設定
    #[serde(default)]
    ai_engine: String,
    #[serde(default)]
    ai_model: String,
    #[serde(default)]
    ai_effort: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ServerStatus {
    running: bool,
    version: String,
    device: String,
    error: String,
    log_path: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct OutputFile {
    name: String,
    path: String,
    size: u64,
    modified: u64,
}

fn home() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| "HOME環境変数が取れへん".to_string())
}

fn app_data(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn settings_file(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data(app)?.join("settings.json"))
}

fn repo_dir() -> Result<PathBuf, String> {
    Ok(home()?.join(IRODORI_REPO))
}

#[tauri::command]
fn load_settings(app: AppHandle) -> Result<Settings, String> {
    let file = settings_file(&app)?;
    let mut s: Settings = match fs::read_to_string(&file) {
        Ok(text) => serde_json::from_str(&text).map_err(|e| e.to_string())?,
        Err(_) => Settings::default(),
    };
    if s.output_dir.is_empty() {
        s.output_dir = home()?.join(DEFAULT_OUT_SUBDIR).to_string_lossy().to_string();
    }
    fs::create_dir_all(&s.output_dir).map_err(|e| format!("保存先フォルダを作れへん: {e}"))?;
    Ok(s)
}

#[tauri::command]
fn save_settings(app: AppHandle, settings: Settings) -> Result<(), String> {
    let file = settings_file(&app)?;
    let text = serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?;
    fs::write(file, text).map_err(|e| e.to_string())
}

#[tauri::command]
fn repo_info() -> Result<serde_json::Value, String> {
    let dir = repo_dir()?;
    Ok(serde_json::json!({
        "repoDir": dir.to_string_lossy(),
        "exists": dir.join("start_server.sh").exists(),
        "serverUrl": SERVER_URL,
    }))
}

fn curl_json(path: &str, timeout_secs: u64) -> Result<serde_json::Value, String> {
    let out = Command::new("curl")
        .args([
            "-s",
            "--max-time",
            &timeout_secs.to_string(),
            &format!("{SERVER_URL}{path}"),
        ])
        .output()
        .map_err(|e| format!("curl を起動できへん: {e}"))?;
    if !out.status.success() {
        return Err("サーバが応答せえへん".to_string());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("サーバの返事が読めへん: {e}"))
}

fn server_log_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data(app)?.join("server.log"))
}

fn status_inner(app: &AppHandle) -> ServerStatus {
    let log_path = server_log_path(app)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    match curl_json("/version", 3) {
        Ok(v) => ServerStatus {
            running: true,
            version: v["version"].as_str().unwrap_or("").to_string(),
            device: v["device"].as_str().unwrap_or("").to_string(),
            error: String::new(),
            log_path,
        },
        Err(e) => ServerStatus {
            running: false,
            version: String::new(),
            device: String::new(),
            error: e,
            log_path,
        },
    }
}

#[tauri::command]
fn server_status(app: AppHandle) -> ServerStatus {
    status_inner(&app)
}

/// サーバが落ちていれば start_server.sh を裏で起動して、応答するまで待つ。
/// モデルのロードと latent 準備で実測16秒前後。初回はモデルDLでもっとかかる。
#[tauri::command(async)]
fn launch_server(app: AppHandle) -> Result<ServerStatus, String> {
    let st = status_inner(&app);
    if st.running {
        return Ok(st);
    }
    let dir = repo_dir()?;
    let script = dir.join("start_server.sh");
    if !script.exists() {
        return Err(format!(
            "いろとりTTSが見つからんかった（{} が無い）。リポジトリを ~/{IRODORI_REPO} に置いてな",
            script.display()
        ));
    }
    // サーバの出力はログに落とす。捨てると「起動したのに応答せえへん」理由が追えなくなる
    let log_path = server_log_path(&app)?;
    let log = fs::File::create(&log_path)
        .map_err(|e| format!("サーバのログを作れへん（{}）: {e}", log_path.display()))?;
    let log2 = log.try_clone().map_err(|e| e.to_string())?;
    let mut cmd = Command::new("/bin/bash");
    cmd.arg(&script)
        .current_dir(&dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log2));
    // 自分のプロセスグループで起動する。アプリを閉じてもサーバは残る（econte とも共用）
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn()
        .map_err(|e| format!("サーバを起動できへん: {e}"))?;

    let mut last = status_inner(&app);
    for _ in 0..180 {
        std::thread::sleep(Duration::from_secs(1));
        last = status_inner(&app);
        if last.running {
            return Ok(last);
        }
    }
    Err(format!(
        "サーバが3分たっても応答せえへん。ログを見てな: {}",
        last.log_path
    ))
}

#[tauri::command]
fn list_outputs(dir: String) -> Result<Vec<OutputFile>, String> {
    let mut files = Vec::new();
    let rd = match fs::read_dir(&dir) {
        Ok(rd) => rd,
        Err(_) => return Ok(files),
    };
    for e in rd.flatten() {
        let p = e.path();
        let is_wav = p
            .extension()
            .and_then(|x| x.to_str())
            .map(|x| x.eq_ignore_ascii_case("wav"))
            .unwrap_or(false);
        let hidden = p
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with('.'))
            .unwrap_or(true);
        if !p.is_file() || !is_wav || hidden {
            continue;
        }
        let meta = match fs::metadata(&p) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        files.push(OutputFile {
            name: p.file_name().unwrap().to_string_lossy().to_string(),
            path: p.to_string_lossy().to_string(),
            size: meta.len(),
            modified,
        });
    }
    files.sort_by(|a, b| b.modified.cmp(&a.modified));
    files.truncate(30);
    Ok(files)
}

#[tauri::command]
fn reveal(path: String) -> Result<(), String> {
    let p = PathBuf::from(&path);
    let status = if p.is_dir() {
        Command::new("open").arg(&p).status()
    } else {
        Command::new("open").args(["-R"]).arg(&p).status()
    }
    .map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err("Finder で開けへんかった".to_string())
    }
}

#[tauri::command]
fn delete_file(path: String) -> Result<(), String> {
    // 消していいのは .wav の実ファイルだけ
    let p = PathBuf::from(&path);
    let is_wav = p
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| x.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);
    if !p.is_file() || !is_wav {
        return Err("wav ファイル以外は消されへん".to_string());
    }
    fs::remove_file(&p).map_err(|e| format!("削除に失敗: {e}"))
}

// ---------------------------------------------------------------------------
// 原稿AI — ローカルの CLI（ChatGPT.app 内蔵 Codex / claude）をヘッドレスで叩く
//
// GUI から起動されたアプリの PATH は /usr/bin:/bin:/usr/sbin:/sbin しか無いので、
// コマンドは必ず絶対パスで解決する（PATH 頼りにすると "not found" で死ぬ）。
// CLAUDE.md の方針どおり「タイムアウト → kill → リトライ」まで面倒を見る。
// ---------------------------------------------------------------------------

const CODEX_CMD: &str = "/Applications/ChatGPT.app/Contents/Resources/codex";
const AI_TIMEOUT_SECS: u64 = 300;

/// claude の実体を探す。PATH が細いので候補を順に見る。
fn claude_path() -> Result<PathBuf, String> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(h) = home() {
        cands.push(h.join(".local/bin/claude"));
        cands.push(h.join(".claude/local/claude"));
        cands.push(h.join(".bun/bin/claude"));
    }
    cands.push(PathBuf::from("/opt/homebrew/bin/claude"));
    cands.push(PathBuf::from("/usr/local/bin/claude"));
    for c in cands {
        if c.is_file() {
            return Ok(c);
        }
    }
    Err("claude コマンドが見つかりません（~/.local/bin/claude などを探しました）".into())
}

/// GUI から起動されたアプリの PATH は細い。CLI が中で使う道具（node など）を
/// 見つけられるように、よくある置き場を前に足しておく。
fn widen_path(cmd: &mut Command) {
    let mut path =
        std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin:/usr/sbin:/sbin".to_string());
    let mut prepend = |dir: PathBuf| {
        if dir.is_dir() {
            path = format!("{}:{}", dir.display(), path);
        }
    };
    for extra in ["/opt/homebrew/bin", "/usr/local/bin"] {
        prepend(PathBuf::from(extra));
    }
    if let Ok(h) = home() {
        for extra in [".bun/bin", ".local/bin"] {
            prepend(h.join(extra));
        }
    }
    cmd.env("PATH", path);
}

/// 子プロセスを待つ。timeout を過ぎたら kill して Err にする（無人で固まらせない）。
fn wait_with_timeout(child: &mut Child, timeout: Duration, label: &str) -> Result<(), String> {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) => {}
            Err(e) => return Err(format!("{label} の待機に失敗: {e}")),
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "{label} が {} 秒たっても終わらへんので打ち切りました",
                timeout.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn tmp_out_path() -> PathBuf {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("irodori_ai_{n}.txt"))
}

/// Codex CLI（ChatGPT.app 内蔵）を1回叩く。思考ログが stdout に混ざるので
/// 最終メッセージだけ -o でファイルに書かせて、それを読む。
fn call_codex(prompt: &str, model: &str, effort: &str) -> Result<String, String> {
    let exe = PathBuf::from(CODEX_CMD);
    if !exe.is_file() {
        return Err(format!(
            "Codex CLI が見つかりません: {CODEX_CMD}（ChatGPT.app を入れてログインしてな）"
        ));
    }
    let out_path = tmp_out_path();
    let mut cmd = Command::new(&exe);
    cmd.arg("exec").arg("--skip-git-repo-check");
    if !model.is_empty() {
        cmd.arg("-m").arg(model);
    }
    if !effort.is_empty() {
        cmd.arg("-c").arg(format!("model_reasoning_effort={effort}"));
    }
    cmd.arg("-o").arg(&out_path).arg("-");
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    widen_path(&mut cmd);

    let mut child = cmd.spawn().map_err(|e| format!("Codex CLI を起動できません: {e}"))?;
    if let Some(mut si) = child.stdin.take() {
        si.write_all(prompt.as_bytes())
            .map_err(|e| format!("Codex CLI にプロンプトを渡せません: {e}"))?;
    }
    let waited = wait_with_timeout(&mut child, Duration::from_secs(AI_TIMEOUT_SECS), "Codex CLI");
    if let Err(e) = waited {
        let _ = fs::remove_file(&out_path);
        return Err(e);
    }
    let text = fs::read_to_string(&out_path).unwrap_or_default();
    let _ = fs::remove_file(&out_path);
    if text.trim().is_empty() {
        return Err("Codex CLI の出力が空でした".into());
    }
    Ok(text)
}

/// claude をヘッドレス（-p）で1回叩く。
fn call_claude(prompt: &str, model: &str, effort: &str) -> Result<String, String> {
    let exe = claude_path()?;
    let mut cmd = Command::new(&exe);
    cmd.arg("-p");
    if !model.is_empty() {
        cmd.arg("--model").arg(model);
    }
    // claude に reasoning effort の引数は無いので、思考トークン量で近づける
    // codex の effort 6段階（none/low/medium/high/xhigh/max）を claude の思考量に寄せる
    let thinking = match effort {
        "none" => "1024",
        "low" => "2048",
        "medium" => "8192",
        "high" => "16384",
        "xhigh" | "max" => "31999",
        _ => "8192",
    };
    cmd.env("MAX_THINKING_TOKENS", thinking);
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    widen_path(&mut cmd);

    let mut child = cmd.spawn().map_err(|e| format!("claude を起動できません: {e}"))?;
    if let Some(mut si) = child.stdin.take() {
        si.write_all(prompt.as_bytes())
            .map_err(|e| format!("claude にプロンプトを渡せません: {e}"))?;
    }
    wait_with_timeout(&mut child, Duration::from_secs(AI_TIMEOUT_SECS), "claude")?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("claude の出力を読めません: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    if text.trim().is_empty() {
        let err = String::from_utf8_lossy(&out.stderr).to_string();
        return Err(format!("claude の出力が空でした: {}", err.chars().take(300).collect::<String>()));
    }
    Ok(text)
}

#[tauri::command]
fn ask_ai(engine: String, model: String, effort: String, prompt: String) -> Result<String, String> {
    if prompt.trim().is_empty() {
        return Err("プロンプトが空です".into());
    }
    let mut last = String::new();
    // 失敗しても2回までやり直す（CLIは時々こける）
    for attempt in 1..=2 {
        let r = if engine == "claude" {
            call_claude(&prompt, &model, &effort)
        } else {
            call_codex(&prompt, &model, &effort)
        };
        match r {
            Ok(text) => return Ok(text),
            Err(e) => {
                last = e;
                if attempt == 1 {
                    std::thread::sleep(Duration::from_millis(800));
                }
            }
        }
    }
    Err(format!("AIの呼び出しに2回失敗しました: {last}"))
}

/// どのCLIが使えるかを画面に出すため
#[tauri::command]
fn ai_available() -> serde_json::Value {
    serde_json::json!({
        "codex": PathBuf::from(CODEX_CMD).is_file(),
        "claude": claude_path().is_ok(),
        "claudePath": claude_path().map(|p| p.display().to_string()).unwrap_or_default(),
    })
}

#[tauri::command]
fn read_log_tail(app: AppHandle) -> Result<String, String> {
    let p = server_log_path(&app)?;
    let text = fs::read_to_string(&p).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(40);
    Ok(lines[start..].join("\n"))
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            load_settings,
            save_settings,
            repo_info,
            server_status,
            launch_server,
            list_outputs,
            reveal,
            delete_file,
            read_log_tail,
            ask_ai,
            ai_available,
        ])
        .run(tauri::generate_context!())
        .expect("いろとりスタジオの起動に失敗");
}
