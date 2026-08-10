/** Preload reserved for future desktop bridges (API key dialog, etc.). */
const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("mashupDesktop", {
  isDesktop: true,
});
