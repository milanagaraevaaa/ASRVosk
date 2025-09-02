const { contextBridge, ipcRenderer } = require('electron');
function getBaseName(fp) {
  return fp.replace(/^.*[\\/]/, '');
}
contextBridge.exposeInMainWorld('electronAPI', {
  selectFile:  ()      => ipcRenderer.invoke('select-file'),
  uploadAudio: (fp)    => ipcRenderer.invoke('upload-audio', fp),
  getBaseName: (fp)    => getBaseName(fp),
});