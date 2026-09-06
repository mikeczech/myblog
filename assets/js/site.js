(function () {
  var root = document.documentElement;
  var picker = document.querySelector(".theme-picker");
  var launcher = document.getElementById("theme-launcher");
  var buttons = document.querySelectorAll("[data-theme-option]");
  var feedFilterButtons = document.querySelectorAll("[data-feed-filter]");
  var feedEntries = document.querySelectorAll("[data-feed-kind]");
  var storageKey = "mike-czech-theme";
  var defaultTheme = root.dataset.theme || "orchid";

  function setTheme(theme) {
    root.dataset.theme = theme;
    buttons.forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.getAttribute("data-theme-option") === theme));
    });
    try {
      localStorage.setItem(storageKey, theme);
    } catch (e) {}
  }

  function setPickerOpen(isOpen) {
    if (!picker || !launcher) {
      return;
    }
    picker.dataset.open = String(isOpen);
    launcher.setAttribute("aria-expanded", String(isOpen));
  }

  setTheme(defaultTheme);

  if (launcher) {
    launcher.addEventListener("click", function () {
      setPickerOpen(!picker || picker.dataset.open !== "true");
    });
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      setTheme(button.getAttribute("data-theme-option"));
      setPickerOpen(false);
    });
  });

  feedFilterButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      var filter = button.getAttribute("data-feed-filter") || "all";
      feedFilterButtons.forEach(function (option) {
        option.setAttribute("aria-pressed", String(option === button));
      });
      feedEntries.forEach(function (entry) {
        var kind = entry.getAttribute("data-feed-kind");
        entry.hidden = filter !== "all" && kind !== filter;
      });
    });
  });

  document.addEventListener("click", function (event) {
    if (picker && !picker.contains(event.target)) {
      setPickerOpen(false);
    }
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      setPickerOpen(false);
    }
  });
})();
