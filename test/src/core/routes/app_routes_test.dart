import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get_it/get_it.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/routes/app_routes.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/ui/pages/chat_page.dart';
import 'package:my_hero/src/features/settings/controller/settings_controller.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/features/settings/ui/pages/settings_screen.dart';
import 'package:my_hero/src/shared/models/model_download_state.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}

class MockMediaPickerService extends Mock implements MediaPickerService {}

class MockSettingsRepository extends Mock implements SettingsRepository {}

void main() {
  late MockChatRepository mockChatRepo;
  late MockMediaPickerService mockPicker;
  late MockSettingsRepository mockSettingsRepo;

  setUp(() async {
    await GetIt.I.reset();
    mockChatRepo = MockChatRepository();
    mockPicker = MockMediaPickerService();
    mockSettingsRepo = MockSettingsRepository();

    when(() => mockChatRepo.getConversations()).thenAnswer((_) async => []);
    when(() => mockChatRepo.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockChatRepo.reset()).thenReturn(null);
    when(() => mockChatRepo.loadOllamaBaseUrl())
        .thenAnswer((_) async => 'http://localhost:11434');
    when(() => mockChatRepo.checkOllamaHealth()).thenAnswer((_) async => true);
    when(() => mockChatRepo.getAvailableModels()).thenAnswer(
      (_) async => [
        const OllamaModel(
          name: 'mimo-v2.6:9b',
          model: 'mimo-v2.6:9b',
          size: 5000000000,
          isLoaded: true,
        ),
      ],
    );

    when(() => mockSettingsRepo.ollamaBaseUrl)
        .thenReturn('http://localhost:11434');
    when(() => mockSettingsRepo.loadOllamaBaseUrl())
        .thenAnswer((_) async => 'http://localhost:11434');
    when(() => mockSettingsRepo.checkOllamaHealth()).thenAnswer((_) async => true);
    when(() => mockSettingsRepo.getAvailableModels()).thenAnswer(
      (_) async => [
        const OllamaModel(
          name: 'mimo-v2.6:9b',
          model: 'mimo-v2.6:9b',
          size: 5000000000,
          isLoaded: true,
        ),
      ],
    );
    when(() => mockSettingsRepo.currentDownloadState)
        .thenReturn(const ModelDownloadState.idle());
    when(() => mockSettingsRepo.downloadStateStream)
        .thenAnswer((_) => const Stream<ModelDownloadState>.empty());

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

  group('AppRoutes GoRouter Tests', () {
    testWidgets('initial route renders ChatPage', (tester) async {
      final router = AppRoutes.createRouter(initialLocation: AppRoutes.initial);

      await tester.pumpWidget(
        MaterialApp.router(
          theme: AppTheme.lightTheme,
          routerConfig: router,
        ),
      );
      await tester.pumpAndSettle();

      expect(find.byType(ChatPage), findsOneWidget);
    });

    testWidgets('/settings route renders SettingsScreen', (tester) async {
      final router = AppRoutes.createRouter(initialLocation: AppRoutes.settings);

      await tester.pumpWidget(
        MaterialApp.router(
          theme: AppTheme.lightTheme,
          routerConfig: router,
        ),
      );
      await tester.pumpAndSettle();

      expect(find.byType(SettingsScreen), findsOneWidget);
      expect(find.text('Configurações da Conversa'), findsOneWidget);
    });

    testWidgets('unknown route renders fallback error screen', (tester) async {
      final router = AppRoutes.createRouter(initialLocation: '/unknown');

      await tester.pumpWidget(
        MaterialApp.router(
          theme: AppTheme.lightTheme,
          routerConfig: router,
        ),
      );
      await tester.pumpAndSettle();

      expect(find.textContaining('Rota não encontrada:'), findsOneWidget);
    });
  });
}
