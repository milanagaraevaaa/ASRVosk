const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const fs = require('fs');

function createWindow() {
  const win = new BrowserWindow({
    width: 800, height: 600,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: __dirname + '/preload.js'
    }
  });
  win.loadFile('index.html');
}

app.whenReady().then(createWindow);

ipcMain.handle('select-file', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Audio', extensions: ['mp3','wav','ogg','flac','aac','m4a','wma','.webm'] }]
  });
  return canceled ? null : filePaths[0];
});

ipcMain.handle('upload-audio', async (event, filePath) => {
  const fileStream = fs.createReadStream(filePath);
  const form = new FormData();
  form.append('file', fileStream);
  const res = await fetch('http://127.0.0.1:8000/transcribe', {
    method: 'POST', body: form
  });
  if (!res.ok) {
    const txt = await res.text();
    let msg = txt;
    try { msg = JSON.parse(txt).detail || msg; } catch {}
    throw new Error(`Server ${res.status}: ${msg}`);
  }
  return await res.json();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});