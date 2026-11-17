import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/core/theme/theme_controller.dart';
import 'package:my_hero/src/features/settings/controller/settings_controller.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/features/settings/ui/pages/settings_screen.dart';
import 'package:my_hero/src/features/settings/ui/widgets/context_window_section.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';
import 'package:my_hero/src/shared/models/model_download_state.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';

class MockSettingsRepository extends Mock implements SettingsRepository {}

void main() {
  late MockSettingsRepository mockRepository;
  late SettingsController controller;

  setUpAll(() {
    registerFallbackValue(ContextMode.automatic);
    registerFallbackValue(ThemeMode.light);
  });

  const modelUnloaded = OllamaModel(
    name: 'mimo-v2.6:9b',
    model: 'mimo-v2.6:9b',
    size: 5000000000,
    parameterSize: '9B',
    isLoaded: false,
  );
  const modelLoaded = OllamaModel(
    name: 'llama3.2:3b',
    model: 'llama3.2:3b',
    size: 2000000000,
    parameterSize: '3.2B',
    isLoaded: true,
    sizeVram: 2500000000,
  );

  setUp(() {
    mockRepository = MockSettingsRepository();
    when(() => mockRepository.ollamaBaseUrl)
        .thenReturn('http://192.168.1.244:11434');
    when(() => mockRepository.loadOllamaBaseUrl())
        .thenAnswer((_) async => 'http://192.168.1.244:11434');
    when(() => mockRepository.saveOllamaBaseUrl(any()))
        .thenAnswer((_) async {});
    when(() => mockRepository.updateOllamaBaseUrl(any())).thenReturn(null);
    when(() => mockRepository.checkOllamaHealth(any()))
        .thenAnswer((_) async => true);
    when(() => mockRepository.checkOllamaHealth())
        .thenAnswer((_) async => true);
    when(() => mockRepository.getAvailableModels())
        .thenAnswer((_) async => [modelUnloaded, modelLoaded]);
    when(() => mockRepository.currentDownloadState)
        .thenReturn(const ModelDownloadState.idle());
    when(() => mockRepository.downloadStateStream)
        .thenAnswer((_) => const Stream<ModelDownloadState>.empty());

    controller = SettingsController(repository: mockRepository);
  });

  Widget buildRoutedTestApp({
    required SettingsArgs args,
    required void Function(SettingsResult? result) onPopped,
  }) {
    final router = GoRouter(
      initialLocation: '/',
      routes: [
        GoRoute(
          path: '/',
          builder: (context, _) => Scaffold(
            body: Center(
              child: ElevatedButton(
                onPressed: () async {
                  final res = await context.push<SettingsResult>('/settings');
                  onPopped(res);
                },
                child: const Text('Open'),
              ),
            ),
          ),
        ),
        GoRoute(
          path: '/settings',
          builder: (context, _) => SettingsScreen(
            initialArgs: args,
            controller: controller,
          ),
        ),
      ],
    );

    return MaterialApp.router(
      theme: AppTheme.lightTheme,
      routerConfig: router,
    );
  }

  group('SettingsScreen Widget Tests', () {
    testWidgets('renders model controls, server section, context section, appearance section, and footer actions without response style section', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.lightTheme,
          home: SettingsScreen(
            initialArgs: const SettingsArgs(
              selectedModelName: 'mimo-v2.6:9b',
              ollamaBaseUrl: 'http://192.168.1.244:11434',
              availableModels: [modelUnloaded, modelLoaded],
              isOllamaConnected: true,
              isModelLoaded: false,
            ),
            controller: controller,
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Configurações da Conversa'), findsOneWidget);
      expect(find.text('Modelo Ollama'), findsOneWidget);
      expect(find.byType(DropdownButtonFormField<String>), findsOneWidget);
      expect(find.text('Carregar na Memória'), findsOneWidget);

      expect(find.text('Estilo de Resposta'), findsNothing);
      expect(find.text('Breve'), findsNothing);
      expect(find.text('Simples'), findsNothing);
      expect(find.text('Detalhada'), findsNothing);
      expect(find.text('Curta'), findsNothing);
      expect(find.text('Padrão'), findsNothing);
      expect(find.text('Longa'), findsNothing);

      expect(find.text('Janela de Contexto'), findsOneWidget);
      expect(find.text('Aparência'), findsOneWidget);
      expect(find.text('Servidor: http://192.168.1.244:11434'), findsOneWidget);
      expect(find.text('Alterar IP'), findsOneWidget);

      expect(find.text('Cancelar'), findsOneWidget);
      expect(find.text('Salvar'), findsOneWidget);
    });

    testWidgets('tapping Alterar IP expands inline editor, tests connection, and applies IP', (tester) async {
      when(() => mockRepository.checkOllamaHealth('http://10.0.0.15:11434'))
          .thenAnswer((_) async => true);

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.lightTheme,
          home: SettingsScreen(
            initialArgs: const SettingsArgs(
              selectedModelName: 'mimo-v2.6:9b',
              ollamaBaseUrl: 'http://192.168.1.244:11434',
              availableModels: [modelUnloaded],
            ),
            controller: controller,
          ),
        ),
      );
      await tester.pumpAndSettle();

      await tester.ensureVisible(find.text('Alterar IP'));
      await tester.tap(find.text('Alterar IP'));
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('server_host_field')), findsOneWidget);
      expect(find.text('Localhost (padrão)'), findsOneWidget);
      expect(find.text('Testar Conexão'), findsOneWidget);
      expect(find.text('Aplicar IP'), findsOneWidget);

      await tester.ensureVisible(find.byKey(const Key('server_host_field')));
      await tester.enterText(find.byKey(const Key('server_host_field')), '10.0.0.15');
      await tester.pump();

      await tester.ensureVisible(find.text('Testar Conexão'));
      await tester.tap(find.text('Testar Conexão'));
      await tester.pumpAndSettle();

      expect(
        find.textContaining('Conectado com sucesso em http://10.0.0.15:11434!'),
        findsOneWidget,
      );

      await tester.ensureVisible(find.text('Aplicar IP'));
      await tester.tap(find.text('Aplicar IP'));
      await tester.pumpAndSettle();

      expect(find.text('Servidor: http://10.0.0.15:11434'), findsOneWidget);
    });

    testWidgets('tapping Carregar na Memória loads model and updates status', (tester) async {
      when(
        () => mockRepository.loadModel(
          modelName: 'mimo-v2.6:9b',
          onProgress: any(named: 'onProgress'),
        ),
      ).thenAnswer((_) async {});
      when(() => mockRepository.getAvailableModels())
          .thenAnswer((_) async => [modelUnloaded.copyWith(isLoaded: true)]);

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.lightTheme,
          home: SettingsScreen(
            initialArgs: const SettingsArgs(
              selectedModelName: 'mimo-v2.6:9b',
              ollamaBaseUrl: 'http://localhost:11434',
              availableModels: [modelUnloaded],
              isModelLoaded: false,
            ),
            controller: controller,
          ),
        ),
      );
      await tester.pumpAndSettle();

      await tester.tap(find.text('Carregar na Memória'));
      await tester.pumpAndSettle();

      expect(find.text('Já carregado na memória do PC'), findsOneWidget);
      expect(find.text('Usar Modelo'), findsOneWidget);
    });

    testWidgets('Cancelar pops returning null and discards unsaved changes', (tester) async {
      SettingsResult? capturedResult = const SettingsResult(
        selectedModelName: 'sentinel',
        ollamaBaseUrl: '',
        availableModels: [],
        isOllamaConnected: false,
        isModelLoaded: false,
      );

      await tester.pumpWidget(
        buildRoutedTestApp(
          args: SettingsArgs(
            selectedModelName: 'mimo-v2.6:9b',
            ollamaBaseUrl: 'http://localhost:11434',
            availableModels: [
              modelUnloaded.copyWith(maxContextLength: 16384),
              modelLoaded,
            ],
          ),
          onPopped: (res) => capturedResult = res,
        ),
      );

      await tester.tap(find.text('Open'));
      await tester.pumpAndSettle();

      await tester.ensureVisible(find.byKey(const Key('context_window_slider')));
      await tester.drag(
        find.byKey(const Key('context_window_slider')),
        const Offset(300, 0),
      );
      await tester.pumpAndSettle();

      await tester.ensureVisible(find.text('Cancelar'));
      await tester.tap(find.text('Cancelar'));
      await tester.pumpAndSettle();

      expect(capturedResult, isNull);
      expect(find.text('Open'), findsOneWidget);
    });

    testWidgets('Salvar pops returning SettingsResult with updated selections', (tester) async {
      SettingsResult? capturedResult;
      when(() => mockRepository.getModelMaxContextLength('llama3.2:3b'))
          .thenAnswer((_) async => 8192);
      when(() => mockRepository.saveContextMode(any())).thenAnswer((_) async {});
      when(() => mockRepository.saveManualContextLength(any()))
          .thenAnswer((_) async {});

      await tester.pumpWidget(
        buildRoutedTestApp(
          args: const SettingsArgs(
            selectedModelName: 'mimo-v2.6:9b',
            ollamaBaseUrl: 'http://localhost:11434',
            availableModels: [modelUnloaded, modelLoaded],
          ),
          onPopped: (res) => capturedResult = res,
        ),
      );

      await tester.tap(find.text('Open'));
      await tester.pumpAndSettle();

      await tester.ensureVisible(find.byType(DropdownButtonFormField<String>));
      await tester.tap(find.byType(DropdownButtonFormField<String>));
      await tester.pumpAndSettle();

      await tester.tap(find.text('llama3.2:3b').last);
      await tester.pumpAndSettle();

      await tester.ensureVisible(find.text('Salvar'));
      await tester.tap(find.text('Salvar'));
      await tester.pumpAndSettle();

      expect(capturedResult, isNotNull);
      expect(capturedResult!.selectedModelName, 'llama3.2:3b');
      expect(find.text('Open'), findsOneWidget);
    });

    testWidgets('context section renders discrete power-of-two slider, header badge, and no legacy mode selector', (tester) async {
      SettingsResult? capturedResult;
      const modelWith16k = OllamaModel(
        name: 'qwen2.5:7b',
        model: 'qwen2.5:7b',
        size: 5000000000,
        isLoaded: true,
        sizeVram: 4500000000,
        maxContextLength: 16384,
      );

      when(() => mockRepository.saveContextMode(any())).thenAnswer((_) async {});
      when(() => mockRepository.saveManualContextLength(any()))
          .thenAnswer((_) async {});

      await tester.pumpWidget(
        buildRoutedTestApp(
          args: const SettingsArgs(
            selectedModelName: 'qwen2.5:7b',
            ollamaBaseUrl: 'http://localhost:11434',
            availableModels: [modelWith16k],
            isModelLoaded: true,
          ),
          onPopped: (res) => capturedResult = res,
        ),
      );

      await tester.tap(find.text('Open'));
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('server_memory_limit_field')), findsNothing);
      expect(find.text('Automático (Dinâmico)'), findsNothing);
      expect(find.text('Manual'), findsNothing);
      expect(find.byType(ChoiceChip), findsNothing);
      expect(find.byType(SegmentedButton<ContextMode>), findsNothing);

      await tester.ensureVisible(find.byKey(const Key('context_window_slider')));

      expect(find.text('4k (Padrão) • 4.096 tokens'), findsOneWidget);
      expect(find.text('4k'), findsOneWidget);
      expect(find.text('8k'), findsNothing);
      expect(find.text('16k'), findsOneWidget);
      expect(find.text('32k'), findsNothing);
      expect(
        find.text('Máximo nativo do modelo ativo: 16.384 tokens'),
        findsOneWidget,
      );

      final sliderTheme = tester.widget<SliderTheme>(
        find.ancestor(
          of: find.byKey(const Key('context_window_slider')),
          matching: find.byType(SliderTheme),
        ).first,
      );
      expect(
        sliderTheme.data.overlayShape,
        const RoundSliderOverlayShape(overlayRadius: 16),
      );
      expect(
        sliderTheme.data.padding,
        const EdgeInsets.symmetric(horizontal: 16),
      );

      final initialSlider = tester.widget<Slider>(
        find.byKey(const Key('context_window_slider')),
      );
      expect(initialSlider.min, 0.0);
      expect(initialSlider.max, 2.0);
      expect(initialSlider.divisions, 2);
      expect(initialSlider.value, 0.0);
      expect(initialSlider.activeColor, AppTheme.lightTheme.colorScheme.primary);
      expect(initialSlider.thumbColor, AppTheme.lightTheme.colorScheme.primary);

      // Tapping the max endpoint jumps to the model limit directly.
      await tester.tap(find.text('16k'));
      await tester.pumpAndSettle();

      expect(find.text('16k • 16.384 tokens'), findsOneWidget);
      final maxSlider = tester.widget<Slider>(
        find.byKey(const Key('context_window_slider')),
      );
      expect(maxSlider.value, 2.0);

      // Tapping the min endpoint jumps back to 4k (4096).
      await tester.tap(find.text('4k'));
      await tester.pumpAndSettle();

      expect(find.text('4k (Padrão) • 4.096 tokens'), findsOneWidget);

      await tester.drag(
        find.byKey(const Key('context_window_slider')),
        const Offset(500, 0),
      );
      await tester.pumpAndSettle();

      expect(find.text('16k • 16.384 tokens'), findsOneWidget);
      final updatedSlider = tester.widget<Slider>(
        find.byKey(const Key('context_window_slider')),
      );
      expect(updatedSlider.value, 2.0);

      await tester.ensureVisible(find.text('Salvar'));
      await tester.tap(find.text('Salvar'));
      await tester.pumpAndSettle();

      expect(capturedResult, isNotNull);
      expect(capturedResult!.contextMode, ContextMode.manual);
      expect(capturedResult!.manualContextLength, 16384);
    });

    testWidgets('renders disabled context slider safely when model supports only 4096 tokens', (tester) async {
      const model4k = OllamaModel(
        name: 'tiny:latest',
        model: 'tiny:latest',
        size: 1000000000,
        isLoaded: true,
        maxContextLength: 4096,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: SettingsScreen(
            initialArgs: const SettingsArgs(
              selectedModelName: 'tiny:latest',
              ollamaBaseUrl: 'http://localhost:11434',
              availableModels: [model4k],
              isOllamaConnected: true,
              isModelLoaded: true,
            ),
            controller: controller,
          ),
        ),
      );
      await tester.pumpAndSettle();

      await tester.ensureVisible(find.byKey(const Key('context_window_slider')));

      expect(find.text('4k (Padrão) • 4.096 tokens'), findsOneWidget);
      expect(find.text('4k'), findsOneWidget);
      expect(
        find.text('Máximo nativo do modelo ativo: 4.096 tokens'),
        findsOneWidget,
      );

      final slider = tester.widget<Slider>(
        find.byKey(const Key('context_window_slider')),
      );
      expect(slider.min, 0.0);
      expect(slider.max, 1.0);
      expect(slider.value, 0.0);
      expect(slider.onChanged, isNull);
    });

    testWidgets('changing model recalculates power-of-two slider steps and clamps selected num_ctx', (tester) async {
      SettingsResult? capturedResult;
      const modelWith64k = OllamaModel(
        name: 'qwen2.5:14b',
        model: 'qwen2.5:14b',
        size: 9000000000,
        isLoaded: false,
        maxContextLength: 65536,
      );

      when(() => mockRepository.getModelMaxContextLength('llama3.2:3b'))
          .thenAnswer((_) async => 8192);
      when(() => mockRepository.saveContextMode(any())).thenAnswer((_) async {});
      when(() => mockRepository.saveManualContextLength(any()))
          .thenAnswer((_) async {});

      await tester.pumpWidget(
        buildRoutedTestApp(
          args: const SettingsArgs(
            selectedModelName: 'qwen2.5:14b',
            ollamaBaseUrl: 'http://localhost:11434',
            availableModels: [modelWith64k, modelLoaded],
            isModelLoaded: false,
            contextMode: ContextMode.manual,
            manualContextLength: 32768,
          ),
          onPopped: (res) => capturedResult = res,
        ),
      );

      await tester.tap(find.text('Open'));
      await tester.pumpAndSettle();

      await tester.ensureVisible(find.byKey(const Key('context_window_slider')));
      expect(find.text('32k • 32.768 tokens'), findsOneWidget);
      expect(find.text('4k'), findsOneWidget);
      expect(find.text('8k'), findsNothing);
      expect(find.text('16k'), findsNothing);
      expect(find.text('32k'), findsNothing);
      expect(find.text('64k'), findsOneWidget);

      final initialSlider = tester.widget<Slider>(
        find.byKey(const Key('context_window_slider')),
      );
      expect(initialSlider.min, 0.0);
      expect(initialSlider.max, 4.0);
      expect(initialSlider.divisions, 4);
      expect(initialSlider.value, 3.0);

      await tester.ensureVisible(find.byType(DropdownButtonFormField<String>));
      await tester.tap(find.byType(DropdownButtonFormField<String>));
      await tester.pumpAndSettle();

      await tester.tap(find.text('llama3.2:3b').last);
      await tester.pumpAndSettle();

      verify(() => mockRepository.getModelMaxContextLength('llama3.2:3b'))
          .called(1);

      await tester.ensureVisible(find.byKey(const Key('context_window_slider')));
      expect(find.text('8k • 8.192 tokens'), findsOneWidget);
      expect(find.text('4k'), findsOneWidget);
      expect(find.text('8k'), findsOneWidget);
      expect(find.text('16k'), findsNothing);
      expect(find.text('32k'), findsNothing);
      expect(find.text('64k'), findsNothing);
      expect(
        find.text('Máximo nativo do modelo ativo: 8.192 tokens'),
        findsOneWidget,
      );

      final clampedSlider = tester.widget<Slider>(
        find.byKey(const Key('context_window_slider')),
      );
      expect(clampedSlider.min, 0.0);
      expect(clampedSlider.max, 1.0);
      expect(clampedSlider.divisions, 1);
      expect(clampedSlider.value, 1.0);

      await tester.ensureVisible(find.text('Salvar'));
      await tester.tap(find.text('Salvar'));
      await tester.pumpAndSettle();

      expect(capturedResult, isNotNull);
      expect(capturedResult!.selectedModelName, 'llama3.2:3b');
      expect(capturedResult!.contextMode, ContextMode.manual);
      expect(capturedResult!.manualContextLength, 8192);
    });

    testWidgets('displays model pull controls, live progress, and cancel button', (tester) async {
      final streamCtrl = StreamController<ModelDownloadState>.broadcast();
      final completer = Completer<ModelDownloadState>();
      when(() => mockRepository.downloadStateStream)
          .thenAnswer((_) => streamCtrl.stream);
      when(() => mockRepository.downloadModel('llama3.2:3b'))
          .thenAnswer((_) => completer.future);
      when(() => mockRepository.cancelModelDownload()).thenAnswer((_) async {
        const cancelled = ModelDownloadState(
          modelName: 'llama3.2:3b',
          status: ModelDownloadStatus.cancelled,
          statusMessage: 'Download de llama3.2:3b cancelado.',
        );
        streamCtrl.add(cancelled);
        if (!completer.isCompleted) {
          completer.complete(cancelled);
        }
      });

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.lightTheme,
          home: SettingsScreen(
            initialArgs: const SettingsArgs(
              selectedModelName: 'mimo-v2.6:9b',
              ollamaBaseUrl: 'http://localhost:11434',
              availableModels: [modelUnloaded],
            ),
            controller: controller,
          ),
        ),
      );
      await tester.pumpAndSettle();

      await tester.enterText(
        find.byKey(const Key('pull_model_name_field')),
        'llama3.2:3b',
      );
      await tester.tap(find.byKey(const Key('pull_model_button')));
      await tester.pump();

      streamCtrl.add(
        const ModelDownloadState(
          modelName: 'llama3.2:3b',
          status: ModelDownloadStatus.downloading,
          statusMessage: 'downloading digest',
          completedBytes: 35,
          totalBytes: 100,
        ),
      );
      await tester.pump();

      expect(find.text('Baixando llama3.2:3b: 35%'), findsOneWidget);
      expect(
        find.byKey(const Key('cancel_model_download_button')),
        findsOneWidget,
      );

      await tester.tap(find.byKey(const Key('cancel_model_download_button')));
      await tester.pumpAndSettle();

      verify(() => mockRepository.cancelModelDownload()).called(1);
      expect(find.text('Download de llama3.2:3b cancelado.'), findsOneWidget);
      await streamCtrl.close();
    });

    testWidgets('appearance section renders Claro/Escuro options and applies theme immediately on click', (tester) async {
      when(() => mockRepository.loadThemeMode())
          .thenAnswer((_) async => ThemeMode.light);
      when(() => mockRepository.saveThemeMode(any())).thenAnswer((_) async {});
      final themeController = ThemeController(repository: mockRepository);

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          themeMode: ThemeMode.light,
          home: SettingsScreen(
            initialArgs: const SettingsArgs(
              selectedModelName: 'mimo-v2.6:9b',
              ollamaBaseUrl: 'http://localhost:11434',
              availableModels: [modelUnloaded],
            ),
            controller: controller,
            themeController: themeController,
          ),
        ),
      );
      await tester.pumpAndSettle();

      await tester.ensureVisible(find.text('Escuro'));
      await tester.pumpAndSettle();
      expect(find.text('Aparência'), findsOneWidget);
      expect(find.text('Claro'), findsOneWidget);
      expect(find.text('Escuro'), findsOneWidget);
      expect(find.byIcon(Icons.light_mode_outlined), findsOneWidget);
      expect(find.byIcon(Icons.dark_mode_outlined), findsOneWidget);

      await tester.tap(find.text('Escuro'));
      await tester.pumpAndSettle();

      expect(themeController.themeMode.value, ThemeMode.dark);
      await tester.ensureVisible(
        find.text('Tema escuro inspirado na paleta Dracula.'),
      );
      expect(find.text('Tema escuro inspirado na paleta Dracula.'), findsOneWidget);
      expect(find.byIcon(Icons.light_mode_outlined), findsOneWidget);
      expect(find.byIcon(Icons.dark_mode_outlined), findsOneWidget);
      verify(() => mockRepository.saveThemeMode(ThemeMode.dark)).called(1);

      await tester.ensureVisible(find.text('Claro'));
      await tester.tap(find.text('Claro'));
      await tester.pumpAndSettle();

      expect(themeController.themeMode.value, ThemeMode.light);
      verify(() => mockRepository.saveThemeMode(ThemeMode.light)).called(1);
    });

    testWidgets('context slider renders 1M and decimal intermediate labels in dark theme and compact layout without overflow', (tester) async {
      await tester.binding.setSurfaceSize(const Size(320, 480));
      addTearDown(() => tester.binding.setSurfaceSize(null));

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          themeMode: ThemeMode.dark,
          home: Scaffold(
            body: Padding(
              padding: const EdgeInsets.all(16),
              child: ContextWindowSection(
                selectedContextLength: 1048576,
                maxModelContext: 1048576,
                onContextLengthChanged: (_) {},
              ),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('1M • 1.048.576 tokens'), findsOneWidget);
      expect(find.text('4k'), findsOneWidget);
      expect(find.text('8k'), findsNothing);
      expect(find.text('512k'), findsNothing);
      expect(find.text('1M'), findsOneWidget);

      final darkSlider = tester.widget<Slider>(
        find.byKey(const Key('context_window_slider')),
      );
      expect(darkSlider.activeColor, AppTheme.darkTheme.colorScheme.primary);
      expect(darkSlider.thumbColor, AppTheme.darkTheme.colorScheme.primary);
      expect(darkSlider.value, 8.0);
      expect(darkSlider.divisions, 8);

      final activeLegend = tester.widget<Text>(find.text('1M'));
      expect(activeLegend.style?.fontWeight, FontWeight.bold);
      expect(activeLegend.style?.color, AppTheme.darkTheme.colorScheme.primary);

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          themeMode: ThemeMode.dark,
          home: Scaffold(
            body: Padding(
              padding: const EdgeInsets.all(16),
              child: ContextWindowSection(
                selectedContextLength: 40000,
                maxModelContext: 40000,
                onContextLengthChanged: (_) {},
              ),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('39.1k • 40.000 tokens'), findsOneWidget);
      expect(find.text('39.1k'), findsOneWidget);
      expect(
        find.text('Máximo nativo do modelo ativo: 40.000 tokens'),
        findsOneWidget,
      );
    });
  });
}
