import 'package:material_ui/material_ui.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/ui/widgets/model_switch_dialog.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}

void main() {
  late MockChatRepository mockRepository;
  late MockMediaPickerService mockMediaPicker;

  setUpAll(() {
    registerFallbackValue(PreferredBackend.cpu);
    registerFallbackValue(ResponseStyle.simple);
  });

  setUp(() {
    mockRepository = MockChatRepository();
    mockMediaPicker = MockMediaPickerService();
    when(() => mockRepository.getConversations()).thenAnswer((_) async => []);
    when(() => mockRepository.getInstalledModelFileNames())
        .thenAnswer((_) async => <String>{});
    when(() => mockRepository.uninstallModel(any())).thenAnswer((_) async {});
    when(() => mockRepository.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockRepository.reset()).thenAnswer((_) async {});
    when(() => mockRepository.activeBackend).thenReturn(PreferredBackend.gpu);
  });

  group('ModelSwitchDialog Widget Tests', () {
    testWidgets('hides NPU on Linux and applies thinking toggle without model reload', (tester) async {
      final controller = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.linux,
      );
      controller.isModelLoaded.value = true;
      controller.loadedModelId.value = AppConstants.model2b.id;

      when(
        () => mockRepository.startNewChatSession(
          modelId: AppConstants.model2b.id,
          contextMessages: any(named: 'contextMessages'),
          responseStyle: any(named: 'responseStyle'),
          enableThinking: true,
        ),
      ).thenAnswer((_) async {});

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: Builder(
              builder: (context) => ElevatedButton(
                onPressed: () => ModelSwitchDialog.show(context, controller),
                child: const Text('Abrir'),
              ),
            ),
          ),
        ),
      );

      await tester.tap(find.text('Abrir'));
      await tester.pumpAndSettle();

      expect(find.text('CPU'), findsOneWidget);
      expect(find.text('GPU'), findsOneWidget);
      expect(find.text('NPU'), findsNothing);
      expect(find.text('32K'), findsOneWidget);

      await tester.ensureVisible(find.text('Modo Pensamento (Thinking)'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Modo Pensamento (Thinking)'));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Salvar'));
      await tester.pumpAndSettle();

      expect(controller.enableThinking.value, isTrue);
      verify(
        () => mockRepository.startNewChatSession(
          modelId: AppConstants.model2b.id,
          contextMessages: any(named: 'contextMessages'),
          responseStyle: ResponseStyle.simple,
          enableThinking: true,
        ),
      ).called(1);
    });

    testWidgets('shows NPU on Windows and reloads model when maxTokens or backend changes', (tester) async {
      final controller = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.windows,
      );
      controller.isModelLoaded.value = true;
      controller.loadedModelId.value = AppConstants.model2b.id;

      when(
        () => mockRepository.loadModel(
          modelId: AppConstants.model2b.id,
          is2B: true,
          preferredBackend: PreferredBackend.npu,
          maxTokens: 16384,
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: ResponseStyle.simple,
          enableThinking: false,
        ),
      ).thenAnswer((_) async {});

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: Builder(
              builder: (context) => ElevatedButton(
                onPressed: () => ModelSwitchDialog.show(context, controller),
                child: const Text('Abrir'),
              ),
            ),
          ),
        ),
      );

      await tester.tap(find.text('Abrir'));
      await tester.pumpAndSettle();

      expect(find.text('NPU'), findsOneWidget);

      await tester.ensureVisible(find.text('NPU'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('NPU'));
      await tester.pumpAndSettle();

      await tester.ensureVisible(find.text('16K'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('16K'));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Salvar'));
      await tester.pumpAndSettle();

      expect(controller.selectedBackend.value, PreferredBackend.npu);
      expect(controller.maxTokens.value, 16384);
      verify(
        () => mockRepository.loadModel(
          modelId: AppConstants.model2b.id,
          is2B: true,
          preferredBackend: PreferredBackend.npu,
          maxTokens: 16384,
          onProgress: any(named: 'onProgress'),
          contextMessages: any(named: 'contextMessages'),
          responseStyle: ResponseStyle.simple,
          enableThinking: false,
        ),
      ).called(1);
    });
  });
}
