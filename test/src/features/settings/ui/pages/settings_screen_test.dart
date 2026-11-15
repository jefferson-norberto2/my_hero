import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/features/settings/controller/settings_controller.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/features/settings/ui/pages/settings_screen.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';
import 'package:my_hero/src/shared/models/response_style.dart';

class MockSettingsRepository extends Mock implements SettingsRepository {}

void main() {
  late MockSettingsRepository mockRepository;
  late SettingsController controller;

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
    testWidgets('renders model controls, response style selector, server section, and footer actions', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.lightTheme,
          home: SettingsScreen(
            initialArgs: const SettingsArgs(
              selectedModelName: 'mimo-v2.6:9b',
              responseStyle: ResponseStyle.simple,
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

      expect(find.text('Estilo de Resposta'), findsOneWidget);
      expect(find.text('Breve'), findsOneWidget);
      expect(find.text('Simples'), findsOneWidget);
      expect(find.text('Detalhada'), findsOneWidget);
      expect(find.text(ResponseStyle.simple.description), findsOneWidget);

      expect(find.text('Servidor: http://192.168.1.244:11434'), findsOneWidget);
      expect(find.text('Alterar IP'), findsOneWidget);

      expect(find.text('Cancelar'), findsOneWidget);
      expect(find.text('Salvar'), findsOneWidget);
    });

    testWidgets('selecting response style updates dynamic description', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.lightTheme,
          home: SettingsScreen(
            initialArgs: const SettingsArgs(
              selectedModelName: 'mimo-v2.6:9b',
              responseStyle: ResponseStyle.simple,
              ollamaBaseUrl: 'http://192.168.1.244:11434',
              availableModels: [modelUnloaded],
            ),
            controller: controller,
          ),
        ),
      );
      await tester.pumpAndSettle();

      await tester.tap(find.text('Detalhada'));
      await tester.pumpAndSettle();

      expect(find.text(ResponseStyle.detailed.description), findsOneWidget);
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
              responseStyle: ResponseStyle.simple,
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

      expect(find.byType(TextField), findsOneWidget);
      expect(find.text('Localhost (padrão)'), findsOneWidget);
      expect(find.text('Testar Conexão'), findsOneWidget);
      expect(find.text('Aplicar IP'), findsOneWidget);

      await tester.ensureVisible(find.byType(TextField));
      await tester.enterText(find.byType(TextField), '10.0.0.15');
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
              responseStyle: ResponseStyle.simple,
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
        responseStyle: ResponseStyle.simple,
        ollamaBaseUrl: '',
        availableModels: [],
        isOllamaConnected: false,
        isModelLoaded: false,
      );

      await tester.pumpWidget(
        buildRoutedTestApp(
          args: const SettingsArgs(
            selectedModelName: 'mimo-v2.6:9b',
            responseStyle: ResponseStyle.simple,
            ollamaBaseUrl: 'http://localhost:11434',
            availableModels: [modelUnloaded, modelLoaded],
          ),
          onPopped: (res) => capturedResult = res,
        ),
      );

      await tester.tap(find.text('Open'));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Breve'));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Cancelar'));
      await tester.pumpAndSettle();

      expect(capturedResult, isNull);
      expect(find.text('Open'), findsOneWidget);
    });

    testWidgets('Salvar pops returning SettingsResult with updated selections', (tester) async {
      SettingsResult? capturedResult;

      await tester.pumpWidget(
        buildRoutedTestApp(
          args: const SettingsArgs(
            selectedModelName: 'mimo-v2.6:9b',
            responseStyle: ResponseStyle.simple,
            ollamaBaseUrl: 'http://localhost:11434',
            availableModels: [modelUnloaded, modelLoaded],
          ),
          onPopped: (res) => capturedResult = res,
        ),
      );

      await tester.tap(find.text('Open'));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Breve'));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Salvar'));
      await tester.pumpAndSettle();

      expect(capturedResult, isNotNull);
      expect(capturedResult!.responseStyle, ResponseStyle.brief);
      expect(capturedResult!.selectedModelName, 'mimo-v2.6:9b');
      expect(find.text('Open'), findsOneWidget);
    });
  });
}
