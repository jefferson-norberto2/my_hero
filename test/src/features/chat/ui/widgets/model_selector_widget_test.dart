import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_model.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/ui/widgets/model_selector_widget.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}

void main() {
  late MockChatRepository mockRepo;
  late MockMediaPickerService mockPicker;
  late ChatController controller;

  setUp(() {
    mockRepo = MockChatRepository();
    mockPicker = MockMediaPickerService();
    controller = ChatController(
      repository: mockRepo,
      mediaPickerService: mockPicker,
    );
  });

  testWidgets('ModelSelectorWidget renders models list without layout crash', (tester) async {
    controller.isOllamaConnected.value = true;
    controller.availableModels.value = [
      const OllamaModel(
        name: 'nemotron-3-nano:4b',
        model: 'nemotron-3-nano:4b',
        size: 2837597147,
        parameterSize: '4.0B',
        isLoaded: true,
      ),
      const OllamaModel(
        name: 'llama3.2:3b',
        model: 'llama3.2:3b',
        size: 2019393189,
        parameterSize: '3.2B',
        isLoaded: false,
      ),
    ];
    controller.selectedModelName.value = 'nemotron-3-nano:4b';

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: ModelSelectorWidget(controller: controller),
          ),
        ),
      ),
    );

    await tester.pumpAndSettle();

    expect(find.byType(DropdownButtonFormField<String>), findsOneWidget);

    // Tap to open the dropdown menu and ensure menu items lay out cleanly
    await tester.tap(find.byType(DropdownButtonFormField<String>));
    await tester.pumpAndSettle();

    expect(find.text('llama3.2:3b').hitTestable(), findsOneWidget);
  });
}
