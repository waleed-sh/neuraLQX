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
