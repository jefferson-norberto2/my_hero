import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/theme/theme_controller.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';

class MockSettingsRepository extends Mock implements SettingsRepository {}

void main() {
  late MockSettingsRepository mockRepository;
  late ThemeController controller;

  setUpAll(() {
    registerFallbackValue(ThemeMode.light);
  });

  setUp(() {
    mockRepository = MockSettingsRepository();
    when(() => mockRepository.loadThemeMode())
        .thenAnswer((_) async => ThemeMode.light);
    when(() => mockRepository.saveThemeMode(any())).thenAnswer((_) async {});
    controller = ThemeController(repository: mockRepository);
  });

  group('ThemeController', () {
    test('defaults to ThemeMode.light', () {
      expect(controller.themeMode.value, ThemeMode.light);
      expect(controller.isDarkMode, isFalse);
    });

    test('loadThemeMode restores persisted ThemeMode.dark from repository', () async {
      when(() => mockRepository.loadThemeMode())
          .thenAnswer((_) async => ThemeMode.dark);

      await controller.loadThemeMode();

      expect(controller.themeMode.value, ThemeMode.dark);
      expect(controller.isDarkMode, isTrue);
    });

    test('loadThemeMode falls back to ThemeMode.light on repository error', () async {
      controller.themeMode.value = ThemeMode.dark;
      when(() => mockRepository.loadThemeMode())
          .thenThrow(Exception('disk error'));

      await controller.loadThemeMode();

      expect(controller.themeMode.value, ThemeMode.light);
      expect(controller.isDarkMode, isFalse);
    });

    test('setThemeMode updates signal immediately and persists preference', () async {
      await controller.setThemeMode(ThemeMode.dark);

      expect(controller.themeMode.value, ThemeMode.dark);
      expect(controller.isDarkMode, isTrue);
      verify(() => mockRepository.saveThemeMode(ThemeMode.dark)).called(1);

      await controller.setThemeMode(ThemeMode.system);
      expect(controller.themeMode.value, ThemeMode.light);
      verify(() => mockRepository.saveThemeMode(ThemeMode.light)).called(1);
    });

    test('toggleTheme switches between light and dark modes', () async {
      await controller.toggleTheme();
      expect(controller.themeMode.value, ThemeMode.dark);
      verify(() => mockRepository.saveThemeMode(ThemeMode.dark)).called(1);

      await controller.toggleTheme();
      expect(controller.themeMode.value, ThemeMode.light);
      verify(() => mockRepository.saveThemeMode(ThemeMode.light)).called(1);
    });
  });
}
