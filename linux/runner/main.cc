#include "my_application.h"

#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <string>
#include <unistd.h>

namespace {

// Returns true if this executable itself is running from inside a Snap package.
bool IsRunningFromSnap() {
  char exe_path[PATH_MAX];
  const ssize_t len =
      readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  if (len <= 0) {
    return false;
  }
  exe_path[len] = '\0';
  return std::strncmp(exe_path, "/snap/", 6) == 0;
}

// Restores an environment variable overridden by VS Code Snap using its
// `<VAR>_VSCODE_SNAP_ORIG` backup. Returns true if restored.
bool RestoreVscodeSnapVar(const char* name) {
  const std::string orig_name = std::string(name) + "_VSCODE_SNAP_ORIG";
  const char* orig_value = std::getenv(orig_name.c_str());
  if (orig_value == nullptr) {
    return false;
  }
  if (orig_value[0] == '\0') {
    unsetenv(name);
  } else {
    setenv(name, orig_value, 1);
  }
  return true;
}

// Removes colon-separated path entries pointing inside a Snap package.
void StripSnapPathsFromListVar(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::strstr(value, "/snap/") == nullptr) {
    return;
  }

  std::string filtered;
  const char* start = value;
  while (*start != '\0') {
    const char* end = std::strchr(start, ':');
    const size_t part_len = (end == nullptr)
                                ? std::strlen(start)
                                : static_cast<size_t>(end - start);
    const std::string part(start, part_len);
    if (!part.empty() && part.find("/snap/") == std::string::npos) {
      if (!filtered.empty()) {
        filtered.push_back(':');
      }
      filtered.append(part);
    }
    if (end == nullptr) {
      break;
    }
    start = end + 1;
  }

  if (filtered.empty()) {
    unsetenv(name);
  } else {
    setenv(name, filtered.c_str(), 1);
  }
}

// Snap-packaged IDEs/terminals (such as VS Code Snap) export GTK/GLib
// environment variables pointing to their older Snap runtime. On newer GNOME
// (e.g. Ubuntu 26.04 / GNOME 50), loading an outdated
// `org.gnome.desktop.interface` schema causes GTK3's Wayland backend to fall
// back to `antialiasing` in the host's
// `org.gnome.settings-daemon.plugins.xsettings` schema (where that key was
// removed), aborting startup with a fatal GLib-GIO-ERROR.
void SanitizeSnapEnvironment() {
  if (IsRunningFromSnap()) {
    return;
  }

  static const char* kSinglePathVars[] = {
      "GSETTINGS_SCHEMA_DIR",
      "GTK_PATH",
      "GTK_EXE_PREFIX",
      "GTK_IM_MODULE_FILE",
      "GIO_MODULE_DIR",
      "LOCPATH",
      "GDK_PIXBUF_MODULE_FILE",
  };

  for (const char* name : kSinglePathVars) {
    const char* value = std::getenv(name);
    if (value != nullptr && std::strstr(value, "/snap/") != nullptr) {
      if (!RestoreVscodeSnapVar(name)) {
        unsetenv(name);
      }
    }
  }

  static const char* kPathListVars[] = {
      "XDG_DATA_DIRS",
      "XDG_CONFIG_DIRS",
  };

  for (const char* name : kPathListVars) {
    const char* value = std::getenv(name);
    if (value != nullptr && std::strstr(value, "/snap/") != nullptr) {
      if (!RestoreVscodeSnapVar(name)) {
        StripSnapPathsFromListVar(name);
      }
    }
  }
}

// Defensive fallback for environments where GTK3 Wayland would still crash due
// to mismatched `org.gnome.desktop.interface` and
// `org.gnome.settings-daemon.plugins.xsettings` schemas.
void EnsureCompatibleGdkBackend() {
  GSettingsSchemaSource* source = g_settings_schema_source_get_default();
  if (source == nullptr) {
    return;
  }

  g_autoptr(GSettingsSchema) interface_schema =
      g_settings_schema_source_lookup(source, "org.gnome.desktop.interface",
                                      TRUE);
  g_autoptr(GSettingsSchema) xsettings_schema =
      g_settings_schema_source_lookup(
          source, "org.gnome.settings-daemon.plugins.xsettings", TRUE);

  if (interface_schema != nullptr &&
      !g_settings_schema_has_key(interface_schema, "font-antialiasing") &&
      xsettings_schema != nullptr &&
      !g_settings_schema_has_key(xsettings_schema, "antialiasing")) {
    setenv("GDK_BACKEND", "x11", 0);
  }
}

}  // namespace

int main(int argc, char** argv) {
  SanitizeSnapEnvironment();
  EnsureCompatibleGdkBackend();

  g_autoptr(MyApplication) app = my_application_new();
  return g_application_run(G_APPLICATION(app), argc, argv);
}
