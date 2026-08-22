// いろとりスタジオ — Irodori-TTS の常駐サーバ(:3952)を使う声のデータベース＆読み上げ保存アプリ
// サーバとの会話（話者追加・合成・レンダー）はフロントの fetch が直接やる。
// Rust 側は「サーバの起動」「設定の保存」「ファイル/Finder まわり」だけを受け持つ。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;
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
        ])
        .run(tauri::generate_context!())
        .expect("いろとりスタジオの起動に失敗");
}
