import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/ui/widgets/model_selector_widget.dart';
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
    testWidgets('shows NPU option on Android and Windows, hides on Linux', (tester) async {
      final androidCtrl = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.android,
      );
      await tester.pumpWidget(wrap(ModelSelectorWidget(controller: androidCtrl)));
      expect(find.text('CPU'), findsOneWidget);
      expect(find.text('GPU'), findsOneWidget);
      expect(find.text('NPU'), findsOneWidget);

      final windowsCtrl = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.windows,
      );
      await tester.pumpWidget(wrap(ModelSelectorWidget(controller: windowsCtrl)));
      expect(find.text('NPU'), findsOneWidget);

      final linuxCtrl = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.linux,
      );
      await tester.pumpWidget(wrap(ModelSelectorWidget(controller: linuxCtrl)));
      expect(find.text('CPU'), findsOneWidget);
      expect(find.text('GPU'), findsOneWidget);
      expect(find.text('NPU'), findsNothing);
    });

    testWidgets('allows selecting maxTokens preset and toggling Thinking Mode', (tester) async {
      final controller = ChatController(
        repository: mockRepository,
        mediaPickerService: mockMediaPicker,
        platform: TargetPlatform.windows,
      );

      await tester.pumpWidget(wrap(ModelSelectorWidget(controller: controller)));

      await tester.tap(find.text('16K'));
      await tester.pump();
      expect(controller.maxTokens.value, 16384);

      await tester.tap(find.text('Modo Pensamento (Thinking)'));
      await tester.pump();
      expect(controller.enableThinking.value, isTrue);

      await tester.tap(find.text('NPU'));
      await tester.pump();
      expect(controller.selectedBackend.value, PreferredBackend.npu);
    });
  });
}
