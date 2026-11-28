import 'package:material_ui/material_ui.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/ui/widgets/model_selector_widget.dart';
import 'package:my_hero/src/shared/services/device_resources_service.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}

void main() {
  late MockChatRepository mockRepository;
  late MockMediaPickerService mockMediaPicker;

  setUp(() {
    mockRepository = MockChatRepository();
    mockMediaPicker = MockMediaPickerService();
    when(() => mockRepository.getConversations()).thenAnswer((_) async => []);
    when(() => mockRepository.getInstalledModelFileNames())
        .thenAnswer((_) async => <String>{});
    when(() => mockRepository.uninstallModel(any())).thenAnswer((_) async {});
    when(() => mockRepository.cleanupStorage()).thenAnswer((_) async {});
    when(() => mockRepository.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockRepository.reset()).thenAnswer((_) async {});
  });

  Widget wrap(Widget child) {
    return MaterialApp(
      home: Scaffold(
        body: SingleChildScrollView(child: child),
      ),
    );
  }

  group('ModelSelectorWidget Tests', () {
    testWidgets('shows backends filtered by model capabilities (CPU+GPU for generic, CPU only for TranslateGemma)', (tester) async {
      final controller = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.android,
      );

      await tester.pumpWidget(wrap(ModelSelectorWidget(controller: controller)));
      expect(find.text('CPU'), findsOneWidget);
      expect(find.text('GPU'), findsOneWidget);
      expect(find.text('NPU'), findsNothing);

      controller.selectModel('translategemma-4b-it');
      await tester.pump();
      expect(find.text('CPU'), findsOneWidget);
      expect(find.text('GPU'), findsNothing);
      expect(find.text('NPU'), findsNothing);
    });

    testWidgets('allows selecting maxTokens preset, toggling Vision and Thinking Mode, and switching backend to GPU', (tester) async {
      final controller = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.windows,
      );

      await tester.pumpWidget(wrap(ModelSelectorWidget(controller: controller)));

      await tester.ensureVisible(find.text('16K'));
      await tester.tap(find.text('16K'));
      await tester.pump();
      expect(controller.maxTokens.value, 16384);

      expect(controller.enableVision.value, isTrue);
      await tester.ensureVisible(find.text('Suporte a imagens (Visão)'));
      await tester.tap(find.text('Suporte a imagens (Visão)'));
      await tester.pump();
      expect(controller.enableVision.value, isFalse);

      await tester.ensureVisible(find.text('Modo Pensamento (Thinking)'));
      await tester.tap(find.text('Modo Pensamento (Thinking)'));
      await tester.pump();
      expect(controller.enableThinking.value, isTrue);

      await tester.ensureVisible(find.text('GPU'));
      await tester.tap(find.text('GPU'));
      await tester.pump();
      expect(controller.selectedBackend.value, PreferredBackend.gpu);
    });

    testWidgets('displays device RAM/storage summary and high memory warning when device RAM is low', (tester) async {
      final controller = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.android,
      );
      controller.deviceResources.value = const DeviceResourcesInfo(
        totalRamBytes: 4 * 1024 * 1024 * 1024,
        availableRamBytes: 1200 * 1024 * 1024,
        lowMemory: false,
        memoryThresholdBytes: 256 * 1024 * 1024,
        freeStorageBytes: 10 * 1024 * 1024 * 1024,
      );
      controller.selectModel(AppConstants.model4b.id);

      await tester.pumpWidget(wrap(ModelSelectorWidget(controller: controller)));

      expect(find.textContaining('Espaço livre:'), findsOneWidget);
      expect(find.textContaining('RAM livre:'), findsAtLeastNWidgets(1));
      expect(find.textContaining('Alto risco de erro de memória'), findsOneWidget);
      expect(find.text('Limpar temporários'), findsOneWidget);
    });

    testWidgets('highlights downloaded models, filters installed and confirms uninstall', (tester) async {
      final gemma2b = AppConstants.model2b;
      final installed = <String>{gemma2b.fileName};

      when(() => mockRepository.getInstalledModelFileNames())
          .thenAnswer((_) async => Set<String>.from(installed));
      when(() => mockRepository.uninstallModel(gemma2b.fileName))
          .thenAnswer((_) async {
        installed.remove(gemma2b.fileName);
      });

      final controller = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.linux,
      );
      await controller.refreshInstalledModels();

      await tester.pumpWidget(wrap(ModelSelectorWidget(controller: controller)));

      expect(find.text('Baixados (1)'), findsOneWidget);
      expect(find.text('Baixado'), findsOneWidget);
      expect(find.text('Carregar ${gemma2b.displayName}'), findsOneWidget);

      // Filtra apenas os baixados
      await tester.tap(find.text('Baixados (1)'));
      await tester.pumpAndSettle();
      expect(find.text(gemma2b.displayName), findsOneWidget);

      // Toca no botão de excluir e confirma no diálogo
      await tester.tap(find.byTooltip('Excluir ${gemma2b.displayName}'));
      await tester.pumpAndSettle();
      expect(find.text('Excluir modelo?'), findsOneWidget);

      await tester.tap(find.text('Excluir'));
      await tester.pumpAndSettle();

      verify(() => mockRepository.uninstallModel(gemma2b.fileName)).called(1);
      expect(find.text('Baixados (0)'), findsOneWidget);
      expect(
        find.text(
          'Baixar e Carregar ${gemma2b.displayName} (${gemma2b.formattedSize})',
        ),
        findsOneWidget,
      );
    });
  });
}
