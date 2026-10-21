import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_input_bar.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}

// 1x1 transparent PNG bytes for Image.memory testing
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
  });

  Widget buildTestWidget() {
    return MaterialApp(
      home: Scaffold(
        body: ChatInputBar(controller: controller),
      ),
    );
  }

  group('ChatInputBar Widget Tests', () {
    testWidgets('renders input bar, photo button and send button', (tester) async {
      await tester.pumpWidget(buildTestWidget());

      expect(find.byType(TextField), findsOneWidget);
      expect(find.byIcon(Icons.add_photo_alternate_rounded), findsOneWidget);
      expect(find.byIcon(Icons.send), findsOneWidget);
    });

    testWidgets('shows attached image preview when attachedImage is set', (tester) async {
      controller.isModelLoaded.value = true;
      controller.setAttachedImage(kTransparentImage);

      await tester.pumpWidget(buildTestWidget());
      await tester.pump();

      expect(find.text('Imagem anexada'), findsOneWidget);
      expect(find.text('Pronta para análise multimodal'), findsOneWidget);
      expect(find.byIcon(Icons.close_rounded), findsOneWidget);
    });

    testWidgets('clears attached image when tapping close icon', (tester) async {
      controller.isModelLoaded.value = true;
      controller.setAttachedImage(kTransparentImage);

      await tester.pumpWidget(buildTestWidget());
      await tester.pump();

      expect(find.text('Imagem anexada'), findsOneWidget);

      await tester.tap(find.byIcon(Icons.close_rounded));
      await tester.pump();

      expect(controller.attachedImage.value, isNull);
      expect(find.text('Imagem anexada'), findsNothing);
    });

    testWidgets('shows bottom sheet modal when tapping photo icon', (tester) async {
      controller.isModelLoaded.value = true;

      await tester.pumpWidget(buildTestWidget());
      await tester.pump();

      await tester.tap(find.byIcon(Icons.add_photo_alternate_rounded));
      await tester.pumpAndSettle();

      expect(find.text('Galeria de fotos'), findsOneWidget);
      expect(find.text('Câmera'), findsOneWidget);
    });
  });
}
