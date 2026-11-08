import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_input_bar.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}

final Uint8List kTransparentImage = Uint8List.fromList(<int>[
  0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49,
  0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06,
  0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44,
  0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D,
  0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42,
  0x60, 0x82,
]);

void main() {
  late MockChatRepository mockRepository;
  late MockMediaPickerService mockMediaPicker;
  late ChatController controller;

  setUpAll(() {
    registerFallbackValue(ResponseStyle.simple);
  });

  setUp(() {
    mockRepository = MockChatRepository();
    mockMediaPicker = MockMediaPickerService();

    when(() => mockRepository.getConversations()).thenAnswer((_) async => []);
    when(() => mockRepository.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockRepository.reset()).thenReturn(null);

    controller = ChatController(
      repository: mockRepository,
      mediaPickerService: mockMediaPicker,
    );
    controller.selectedModelName.value = 'llama3.2:3b';
  });

  Widget buildTestWidget() {
    return MaterialApp(
      home: Scaffold(
        body: ChatInputBar(controller: controller),
      ),
    );
  }

  group('ChatInputBar Widget Tests', () {
    testWidgets('renders input bar with multiline support, attachment button and send button', (tester) async {
      await tester.pumpWidget(buildTestWidget());

      final textFieldFinder = find.byType(TextField);
      expect(textFieldFinder, findsOneWidget);
      expect(find.byIcon(Icons.attach_file_rounded), findsOneWidget);
      expect(find.byIcon(Icons.send), findsOneWidget);

      final textField = tester.widget<TextField>(textFieldFinder);
      expect(textField.minLines, 2);
      expect(textField.maxLines, 10);
      expect(textField.keyboardType, TextInputType.multiline);
    });

    testWidgets('allows typing multiline text with newlines', (tester) async {
      controller.isModelLoaded.value = true;
      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
          attachment: any(named: 'attachment'),
          modelName: any(named: 'modelName'),
          history: any(named: 'history'),
          responseStyle: any(named: 'responseStyle'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['Resposta']));

      await tester.pumpWidget(buildTestWidget());
      await tester.pump();

      final multilineText = 'void main() {\n  print("hello world");\n}';
      await tester.enterText(find.byType(TextField), multilineText);
      await tester.pump();

      expect(find.text(multilineText), findsOneWidget);

      await tester.tap(find.byIcon(Icons.send));
      await tester.pump();

      expect(find.text(multilineText), findsNothing);
      expect(controller.chatHistory.value.first.text, multilineText);
    });

    testWidgets('shows attached image preview when attachment is set', (tester) async {
      controller.isModelLoaded.value = true;
      controller.setAttachedImage(kTransparentImage);

      await tester.pumpWidget(buildTestWidget());
      await tester.pump();

      expect(find.text('imagem.png'), findsOneWidget);
      expect(find.byIcon(Icons.close_rounded), findsOneWidget);
    });

    testWidgets('shows audio attachment preview and clears it', (tester) async {
      controller.isModelLoaded.value = true;
      controller.setAttachment(
        ChatAttachment(
          name: 'audio.mp3',
          type: AttachmentType.audio,
          bytes: Uint8List.fromList([1, 2, 3, 4]),
        ),
      );

      await tester.pumpWidget(buildTestWidget());
      await tester.pump();

      expect(find.text('audio.mp3'), findsOneWidget);
      expect(find.byIcon(Icons.audiotrack_rounded), findsOneWidget);

      await tester.tap(find.byIcon(Icons.close_rounded));
      await tester.pump();

      expect(controller.attachedAttachment.value, isNull);
      expect(find.text('audio.mp3'), findsNothing);
    });

    testWidgets('shows document attachment preview', (tester) async {
      controller.isModelLoaded.value = true;
      controller.setAttachment(
        ChatAttachment(
          name: 'codigo.dart',
          type: AttachmentType.document,
          bytes: Uint8List.fromList([10, 20]),
          textContent: 'void main() {}',
        ),
      );

      await tester.pumpWidget(buildTestWidget());
      await tester.pump();

      expect(find.text('codigo.dart'), findsOneWidget);
      expect(find.byIcon(Icons.description_rounded), findsOneWidget);
    });

    testWidgets('shows bottom sheet modal with image, audio and document when tapping attachment icon', (tester) async {
      controller.isModelLoaded.value = true;

      await tester.pumpWidget(buildTestWidget());
      await tester.pump();

      await tester.tap(find.byIcon(Icons.attach_file_rounded));
      await tester.pumpAndSettle();

      expect(find.text('Adicionar Anexo'), findsOneWidget);
      expect(find.text('Imagem'), findsOneWidget);
      expect(find.text('Áudio'), findsOneWidget);
      expect(find.text('Documento ou Código'), findsOneWidget);
    });
  });
}
