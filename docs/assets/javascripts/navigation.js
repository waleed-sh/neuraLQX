/* Keep the mobile menu's focus and expanded state in sync with its dialog. */
(() => {
  const dialog = document.getElementById("bottom-sidebar");
  const trigger = document.getElementById("burger-button");
  if (!dialog || !trigger) return;

  trigger.addEventListener("click", () => {
    if (dialog.open) {
      dialog.close();
    } else {
      dialog.showModal();
      trigger.setAttribute("aria-expanded", "true");
    }
  });
  dialog.addEventListener("close", () => trigger.setAttribute("aria-expanded", "false"));
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.querySelector("[data-close-navigation]").addEventListener("click", () => dialog.close());
  window.matchMedia("(min-width: 1024px)").addEventListener("change", (event) => {
    if (event.matches && dialog.open) dialog.close();
  });
})();

/* Preserve the nested sidebar position and reveal the active page only as needed. */
(() => {
  const sidebar = document.querySelector('[data-sidebar="content"]')
  if (!sidebar) return
  const storageKey = "nqx-sidebar-scroll"

  const restorePosition = () => {
    if (!sidebar.clientHeight) return
    try {
      const saved = sessionStorage.getItem(storageKey)
      const offset = Number(saved)
      if (saved !== null && Number.isFinite(offset) && offset >= 0) {
        sidebar.scrollTop = offset
      }
    } catch {
      // Navigation still works when browser storage is unavailable.
    }

    const active = sidebar.querySelector('a[aria-current="page"]')
    if (!active) return
    const viewport = sidebar.getBoundingClientRect()
    const link = active.getBoundingClientRect()
    const padding = 12
    if (link.top < viewport.top + padding) {
      sidebar.scrollTop += link.top - viewport.top - padding
    } else if (link.bottom > viewport.bottom - padding) {
      sidebar.scrollTop += link.bottom - viewport.bottom + padding
    }
  }

  window.addEventListener("pageshow", (event) => {
    if (!event.persisted) restorePosition()
  })
  window.addEventListener("pagehide", () => {
    if (!sidebar.clientHeight) return
    try {
      sessionStorage.setItem(storageKey, String(sidebar.scrollTop))
    } catch {
      // Saving the position is optional.
    }
  })
  window.matchMedia("(min-width: 1024px)").addEventListener("change", (event) => {
    if (event.matches) restorePosition()
  })
})();

/* Restore fragment navigation after the theme finishes formatting API content. */
window.addEventListener("load", () => {
  if (!window.location.hash) return
  const target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)))
  target?.scrollIntoView({ block: "start", behavior: "instant" })
})
