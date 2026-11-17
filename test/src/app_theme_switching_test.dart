import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get_it/get_it.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/app.dart';
import 'package:my_hero/src/core/routes/app_routes.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/core/theme/theme_controller.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/ui/pages/chat_page.dart';
import 'package:my_hero/src/features/settings/controller/settings_controller.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/features/settings/ui/pages/settings_screen.dart';
import 'package:my_hero/src/shared/models/model_download_state.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';
import 'package:my_hero/src/shared/models/response_style.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}

class MockMediaPickerService extends Mock implements MediaPickerService {}

class MockSettingsRepository extends Mock implements SettingsRepository {}

void main() {
  late MockChatRepository mockChatRepo;
  late MockMediaPickerService mockPicker;
  late MockSettingsRepository mockSettingsRepo;
  late ThemeController themeController;

  const loadedModel = OllamaModel(
    name: 'qwen2.5:7b',
    model: 'qwen2.5:7b',
    size: 4000000000,
    isLoaded: true,
    sizeVram: 4000000000,
    contextLength: 8192,
  );

  setUpAll(() {
    registerFallbackValue(ResponseStyle.simple);
    registerFallbackValue(ThemeMode.light);
  });

  setUp(() async {
    await GetIt.I.reset();
    mockChatRepo = MockChatRepository();
    mockPicker = MockMediaPickerService();
    mockSettingsRepo = MockSettingsRepository();

    final now = DateTime(2026, 3, 1);
    final conversation = Conversation(
      id: 'theme-conv-1',
      title: 'Tema Teste',
      modelName: 'qwen2.5:7b',
      messages: [
        ChatMessage(content: 'Olá', isUser: true),
        ChatMessage(
          content: 'Oi! Tudo certo.',
          thinkingContent: 'Raciocínio',
          isUser: false,
        ),
      ],
      createdAt: now,
      updatedAt: now,
    );

    when(() => mockChatRepo.getConversations())
        .thenAnswer((_) async => [conversation]);
    when(() => mockChatRepo.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockChatRepo.reset()).thenReturn(null);
    when(() => mockChatRepo.loadOllamaBaseUrl())
        .thenAnswer((_) async => 'http://localhost:11434');
    when(() => mockChatRepo.checkOllamaHealth()).thenAnswer((_) async => true);
    when(() => mockChatRepo.getAvailableModels())
        .thenAnswer((_) async => [loadedModel]);

    when(() => mockSettingsRepo.ollamaBaseUrl)
        .thenReturn('http://localhost:11434');
    when(() => mockSettingsRepo.loadOllamaBaseUrl())
        .thenAnswer((_) async => 'http://localhost:11434');
    when(() => mockSettingsRepo.checkOllamaHealth())
        .thenAnswer((_) async => true);
    when(() => mockSettingsRepo.getAvailableModels())
        .thenAnswer((_) async => [loadedModel]);
    when(() => mockSettingsRepo.currentDownloadState)
        .thenReturn(const ModelDownloadState.idle());
    when(() => mockSettingsRepo.downloadStateStream)
        .thenAnswer((_) => const Stream<ModelDownloadState>.empty());
    when(() => mockSettingsRepo.loadThemeMode())
        .thenAnswer((_) async => ThemeMode.light);
    when(() => mockSettingsRepo.saveThemeMode(any())).thenAnswer((_) async {});

    themeController = ThemeController(repository: mockSettingsRepo);
    GetIt.I.registerSingleton<ThemeController>(themeController);
    GetIt.I.registerFactory<ChatController>(
      () => ChatController(
        repository: mockChatRepo,
        mediaPickerService: mockPicker,
      ),
    );
    GetIt.I.registerFactory<SettingsController>(
      () => SettingsController(repository: mockSettingsRepo),
    );
  });

  tearDown(() async {
    await GetIt.I.reset();
  });

  group('GemmaChatApp Theme Switching & Dracula Palette', () {
    testWidgets('renders initial frame in Dracula dark theme when preloaded as dark', (tester) async {
      when(() => mockSettingsRepo.loadThemeMode())
          .thenAnswer((_) async => ThemeMode.dark);
      await themeController.loadThemeMode();

      final router = AppRoutes.createRouter(initialLocation: AppRoutes.initial);

      await tester.pumpWidget(
        GemmaChatApp(
          themeController: themeController,
          routerConfig: router,
        ),
      );
      await tester.pumpAndSettle();

      final chatContext = tester.element(find.byType(ChatPage));
      final theme = Theme.of(chatContext);
      final appColors = AppThemeColors.of(chatContext);

      expect(theme.brightness, Brightness.dark);
      expect(theme.scaffoldBackgroundColor, const Color(0xFF282A36));
      expect(theme.appBarTheme.backgroundColor, const Color(0xFF21222C));
      expect(theme.colorScheme.primary, const Color(0xFFBD93F9));
      expect(theme.colorScheme.onPrimary, const Color(0xFF282A36));
      expect(appColors.linkColor, const Color(0xFF8BE9FD));
      expect(appColors.success, const Color(0xFF50FA7B));
    });

    testWidgets('switching theme in SettingsScreen updates MaterialApp.router immediately without resetting GoRouter stack', (tester) async {
      final router = AppRoutes.createRouter(initialLocation: AppRoutes.initial);

      await tester.pumpWidget(
        GemmaChatApp(
          themeController: themeController,
          routerConfig: router,
        ),
      );
      await tester.pumpAndSettle();

      // Starts in light mode by default.
      expect(
        Theme.of(tester.element(find.byType(ChatPage))).brightness,
        Brightness.light,
      );

      // Navigate to SettingsScreen via GoRouter.
      await tester.tap(find.byIcon(Icons.tune));
      await tester.pumpAndSettle();
      expect(find.byType(SettingsScreen), findsOneWidget);

      // Select Escuro in Aparência section.
      await tester.ensureVisible(find.text('Escuro'));
      await tester.tap(find.text('Escuro'));
      await tester.pumpAndSettle();

      // Must remain on SettingsScreen (GoRouter stack preserved) and be dark Dracula.
      expect(find.byType(SettingsScreen), findsOneWidget);
      final settingsTheme = Theme.of(tester.element(find.byType(SettingsScreen)));
      expect(settingsTheme.brightness, Brightness.dark);
      expect(settingsTheme.scaffoldBackgroundColor, const Color(0xFF282A36));
      expect(settingsTheme.colorScheme.primary, const Color(0xFFBD93F9));
      verify(() => mockSettingsRepo.saveThemeMode(ThemeMode.dark)).called(1);

      // Pop back to ChatPage and confirm ChatPage also reflects Dracula dark mode.
      await tester.tap(find.byIcon(Icons.arrow_back));
      await tester.pumpAndSettle();

      expect(find.byType(ChatPage), findsOneWidget);
      final chatTheme = Theme.of(tester.element(find.byType(ChatPage)));
      expect(chatTheme.brightness, Brightness.dark);
      expect(chatTheme.scaffoldBackgroundColor, const Color(0xFF282A36));
    });
  });
}
