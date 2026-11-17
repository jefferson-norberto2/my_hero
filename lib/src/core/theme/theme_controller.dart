import 'package:flutter/material.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:signals_flutter/signals_flutter.dart';

/// Manages the application's active [ThemeMode] reactively and persists user
/// appearance preferences in local storage.
class ThemeController {
  final SettingsRepository _repository;

  ThemeController({required SettingsRepository repository})
      : _repository = repository;

  /// Reactive theme mode defaulting to light theme.
  final themeMode = signal<ThemeMode>(ThemeMode.light);

  /// Whether the dark theme (Dracula) is currently active.
  bool get isDarkMode => themeMode.value == ThemeMode.dark;

  /// Loads the saved theme preference before the initial app render.
  Future<void> loadThemeMode() async {
    try {
      themeMode.value = await _repository.loadThemeMode();
    } catch (_) {
      themeMode.value = ThemeMode.light;
    }
  }

  /// Applies [mode] immediately and persists it to local storage.
  Future<void> setThemeMode(ThemeMode mode) async {
    final normalizedMode =
        mode == ThemeMode.dark ? ThemeMode.dark : ThemeMode.light;
    if (themeMode.value != normalizedMode) {
      themeMode.value = normalizedMode;
    }

    try {
      await _repository.saveThemeMode(normalizedMode);
    } catch (_) {
      // Keep reactive in-memory theme applied even if disk write fails.
    }
  }

  /// Toggles between light and dark theme modes.
  Future<void> toggleTheme() {
    final nextMode = isDarkMode ? ThemeMode.light : ThemeMode.dark;
    return setThemeMode(nextMode);
  }
}
