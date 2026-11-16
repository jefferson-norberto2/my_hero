import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/routes/app_routes.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_memory_parser.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/ui/pages/chat_page.dart';
import 'package:my_hero/src/features/settings/controller/settings_controller.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
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
  late ChatController chatController;
  late SettingsController settingsController;

  const loadedModel = OllamaModel(
    name: 'mimo-v2.6:9b',
    model: 'mimo-v2.6:9b',
    size: 5000000000,
    parameterSize: '9B',
    isLoaded: true,
    sizeVram: 4080218931, // ~3.8 GB
    contextLength: 8192,
  );

  const unloadedModel = OllamaModel(
    name: 'mimo-v2.6:9b',
    model: 'mimo-v2.6:9b',
    size: 5000000000,
    parameterSize: '9B',
    isLoaded: false,
  );

  setUpAll(() {
    registerFallbackValue(ResponseStyle.simple);
  });

  setUp(() {
    mockChatRepo = MockChatRepository();
    mockPicker = MockMediaPickerService();
    mockSettingsRepo = MockSettingsRepository();

    when(() => mockChatRepo.getConversations()).thenAnswer((_) async => []);
    when(() => mockChatRepo.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockChatRepo.reset()).thenReturn(null);
    when(() => mockChatRepo.loadOllamaBaseUrl())
        .thenAnswer((_) async => 'http://192.168.1.244:11434');
    when(() => mockChatRepo.checkOllamaHealth()).thenAnswer((_) async => true);
    when(() => mockChatRepo.updateOllamaBaseUrl(any())).thenReturn(null);
    when(() => mockChatRepo.saveOllamaBaseUrl(any())).thenAnswer((_) async {});

    when(() => mockSettingsRepo.ollamaBaseUrl)
        .thenReturn('http://192.168.1.244:11434');
    when(() => mockSettingsRepo.loadOllamaBaseUrl())
        .thenAnswer((_) async => 'http://192.168.1.244:11434');
    when(() => mockSettingsRepo.updateOllamaBaseUrl(any())).thenReturn(null);
    when(() => mockSettingsRepo.saveOllamaBaseUrl(any()))
        .thenAnswer((_) async {});
    when(() => mockSettingsRepo.checkOllamaHealth(any()))
        .thenAnswer((_) async => true);
    when(() => mockSettingsRepo.checkOllamaHealth())
        .thenAnswer((_) async => true);
    when(() => mockSettingsRepo.getAvailableModels())
        .thenAnswer((_) async => [loadedModel]);
    when(() => mockSettingsRepo.currentDownloadState)
        .thenReturn(const ModelDownloadState.idle());
    when(() => mockSettingsRepo.downloadStateStream)
        .thenAnswer((_) => const Stream<ModelDownloadState>.empty());

    chatController = ChatController(
      repository: mockChatRepo,
      mediaPickerService: mockPicker,
    );
    settingsController = SettingsController(repository: mockSettingsRepo);
  });

  Widget buildApp() {
    final router = GoRouter(
      initialLocation: AppRoutes.initial,
      routes: [
        GoRoute(
          path: AppRoutes.initial,
          builder: (context, state) => ChatPage(controller: chatController),
        ),
        GoRoute(
          path: AppRoutes.settings,
          builder: (context, state) => SettingsScreen(
            initialArgs: state.extra as SettingsArgs?,
            controller: settingsController,
          ),
        ),
      ],
    );

    return MaterialApp.router(
      theme: AppTheme.lightTheme,
      routerConfig: router,
    );
  }

  group('ChatPage Layout & AppBar Tests', () {
    testWidgets('hides top ModelSelectorWidget card and shows unloaded status in AppBar subtitle', (tester) async {
      when(() => mockChatRepo.getAvailableModels())
          .thenAnswer((_) async => [unloadedModel]);

      await tester.pumpWidget(buildApp());
      await tester.pumpAndSettle();

      // Top model selector card should no longer exist in ChatPage.
      expect(find.text('Modelos disponíveis no seu Ollama:'), findsNothing);
      expect(find.text('Carregar na Memória'), findsNothing);

      // AppBar subtitle shows unloaded status.
      expect(
        find.text('mimo-v2.6:9b (não carregado na memória)'),
        findsOneWidget,
      );

      // AppBar actions have only the single settings sliders icon.
      expect(find.byIcon(Icons.tune), findsOneWidget);
      expect(find.byIcon(Icons.dns_rounded), findsNothing);
      expect(find.byIcon(Icons.refresh), findsNothing);
    });

    testWidgets('shows model and style without duplicate VRAM in AppBar subtitle, keeping VRAM and Ctx exclusively in right pill', (tester) async {
      when(() => mockChatRepo.getAvailableModels())
          .thenAnswer((_) async => [loadedModel]);

      await tester.pumpWidget(buildApp());
      await tester.pumpAndSettle();

      expect(find.text('Nova Conversa'), findsOneWidget);
      expect(find.text('mimo-v2.6:9b • Simples'), findsOneWidget);
      expect(find.textContaining('VRAM:'), findsNothing);

      expect(find.text('3.8 GB'), findsOneWidget);
      expect(find.text('GPU'), findsOneWidget);
      expect(find.byTooltip('VRAM: 3.8 GB | RAM: 0.0 GB'), findsOneWidget);
      expect(find.text('Ctx: 0% (restam 100%)'), findsOneWidget);

      expect(find.byIcon(Icons.tune), findsOneWidget);
      expect(find.byIcon(Icons.dns_rounded), findsNothing);
      expect(find.byIcon(Icons.refresh), findsNothing);
    });

    testWidgets('shows CPU memory badge tag, tooltip, and metrics dialog breakdown', (tester) async {
      final cpuModel = OllamaModel(
        name: 'llama3.2:latest',
        model: 'llama3.2:latest',
        size: 5000000000,
        isLoaded: true,
        sizeVram: 0,
        memoryAllocation: OllamaMemoryParser.fromBytes(
          sizeBytes: 5798205849,
          vramBytes: 0,
        ),
      );
      when(() => mockChatRepo.getAvailableModels())
          .thenAnswer((_) async => [cpuModel]);

      await tester.pumpWidget(buildApp());
      await tester.pumpAndSettle();

      expect(find.text('5.4 GB'), findsOneWidget);
      expect(find.text('CPU'), findsOneWidget);
      expect(find.byTooltip('Memória RAM (CPU): 5.4 GB'), findsOneWidget);

      await tester.tap(find.text('5.4 GB'));
      await tester.pumpAndSettle();

      expect(find.text('Memória Principal (RAM / CPU)'), findsOneWidget);
      expect(find.text('Alocado no Ollama: 5.4 GB'), findsOneWidget);
      expect(find.text('Memória RAM (CPU): 5.4 GB'), findsOneWidget);
    });

    testWidgets('shows GPU/CPU tag in AppBar badge for hybrid execution mode', (tester) async {
      final hybridModel = OllamaModel(
        name: 'qwen3:8b',
        model: 'qwen3:8b',
        size: 5000000000,
        isLoaded: true,
        sizeVram: 3221225472,
        memoryAllocation: OllamaMemoryParser.fromBytes(
          sizeBytes: 5798205849,
          vramBytes: 3221225472,
        ),
      );
      when(() => mockChatRepo.getAvailableModels())
          .thenAnswer((_) async => [hybridModel]);

      await tester.pumpWidget(buildApp());
      await tester.pumpAndSettle();

      expect(find.text('5.4 GB'), findsOneWidget);
      expect(find.text('GPU/CPU'), findsOneWidget);
      expect(find.byTooltip('VRAM: 3.0 GB | RAM: 2.4 GB'), findsOneWidget);
    });

    testWidgets('navigates to SettingsScreen via GoRouter and updates ChatPage on Salvar', (tester) async {
      when(() => mockChatRepo.getAvailableModels())
          .thenAnswer((_) async => [loadedModel]);

      await tester.pumpWidget(buildApp());
      await tester.pumpAndSettle();

      await tester.tap(find.byIcon(Icons.tune));
      await tester.pumpAndSettle();

      expect(find.text('Configurações da Conversa'), findsOneWidget);

      await tester.tap(find.text('Breve'));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Salvar'));
      await tester.pumpAndSettle();

      expect(find.text('mimo-v2.6:9b • Breve'), findsOneWidget);
    });

    testWidgets('navigates to SettingsScreen and discards changes on Cancelar', (tester) async {
      when(() => mockChatRepo.getAvailableModels())
          .thenAnswer((_) async => [loadedModel]);

      await tester.pumpWidget(buildApp());
      await tester.pumpAndSettle();

      await tester.tap(find.byIcon(Icons.tune));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Detalhada'));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Cancelar'));
      await tester.pumpAndSettle();

      expect(find.text('mimo-v2.6:9b • Simples'), findsOneWidget);
    });

    testWidgets('drawer settings entry navigates to SettingsScreen', (tester) async {
      when(() => mockChatRepo.getAvailableModels())
          .thenAnswer((_) async => [loadedModel]);

      await tester.pumpWidget(buildApp());
      await tester.pumpAndSettle();

      await tester.tap(find.byTooltip('Open navigation menu'));
      await tester.pumpAndSettle();

      expect(find.text('Configurações'), findsOneWidget);
      await tester.tap(find.text('Configurações'));
      await tester.pumpAndSettle();

      expect(find.text('Configurações da Conversa'), findsOneWidget);
    });
  });
}
