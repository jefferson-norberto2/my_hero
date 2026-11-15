import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/ui/widgets/ollama_server_config_dialog.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}

void main() {
  setUpAll(() {
    registerFallbackValue(ResponseStyle.simple);
  });

  late MockChatRepository mockRepository;
  late MockMediaPickerService mockMediaPicker;
  late ChatController controller;

  setUp(() {
    mockRepository = MockChatRepository();
    mockMediaPicker = MockMediaPickerService();

    when(() => mockRepository.getConversations()).thenAnswer((_) async => []);
    when(() => mockRepository.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockRepository.reset()).thenReturn(null);
    when(() => mockRepository.checkOllamaHealth(any())).thenAnswer((_) async => true);
    when(() => mockRepository.checkOllamaHealth()).thenAnswer((_) async => true);
    when(() => mockRepository.loadOllamaBaseUrl()).thenAnswer((_) async => 'http://localhost:11434');
    when(() => mockRepository.updateOllamaBaseUrl(any())).thenReturn(null);
    when(() => mockRepository.saveOllamaBaseUrl(any())).thenAnswer((_) async {});
    when(() => mockRepository.getAvailableModels()).thenAnswer((_) async => [
      const OllamaModel(
        name: 'llama3.2:3b',
        model: 'llama3.2:3b',
        size: 2000000000,
        parameterSize: '3.2B',
      ),
    ]);

    controller = ChatController(
      repository: mockRepository,
      mediaPickerService: mockMediaPicker,
    );
  });

  Widget buildTestableWidget(Widget child) {
    return MaterialApp(
      home: Scaffold(
        body: child,
      ),
    );
  }

  testWidgets('OllamaServerConfigDialog renders input field, tip card and buttons', (tester) async {
    await tester.pumpWidget(
      buildTestableWidget(
        OllamaServerConfigDialog(controller: controller),
      ),
    );

    expect(find.text('Servidor Ollama na Rede'), findsOneWidget);
    expect(find.byType(TextField), findsOneWidget);
    expect(find.text('Localhost (padrão)'), findsOneWidget);
    expect(find.text('Testar Conexão'), findsOneWidget);
    expect(find.text('Salvar e Conectar'), findsOneWidget);
    expect(find.textContaining('OLLAMA_HOST=0.0.0.0'), findsOneWidget);
  });

  testWidgets('OllamaServerConfigDialog clicking Localhost chip updates input text', (tester) async {
    await tester.pumpWidget(
      buildTestableWidget(
        OllamaServerConfigDialog(controller: controller),
      ),
    );

    final textFieldFinder = find.byType(TextField);
    await tester.enterText(textFieldFinder, '192.168.1.99');
    await tester.pump();
    expect(find.text('192.168.1.99'), findsOneWidget);

    await tester.tap(find.text('Localhost (padrão)'));
    await tester.pump();

    expect(find.text('http://localhost:11434'), findsOneWidget);
  });

  testWidgets('OllamaServerConfigDialog testing connection shows success message', (tester) async {
    when(() => mockRepository.checkOllamaHealth('http://192.168.1.50:11434'))
        .thenAnswer((_) async => true);

    await tester.pumpWidget(
      buildTestableWidget(
        OllamaServerConfigDialog(controller: controller),
      ),
    );

    final textFieldFinder = find.byType(TextField);
    await tester.enterText(textFieldFinder, '192.168.1.50');
    await tester.pump();

    await tester.tap(find.text('Testar Conexão'));
    await tester.pump(); // Start testing
    await tester.pump(const Duration(milliseconds: 100)); // Finish testing

    expect(find.textContaining('Conectado com sucesso em http://192.168.1.50:11434!'), findsOneWidget);
  });

  testWidgets('OllamaServerConfigDialog saving calls controller and updates host', (tester) async {
    await tester.pumpWidget(
      buildTestableWidget(
        OllamaServerConfigDialog(controller: controller),
      ),
    );

    final textFieldFinder = find.byType(TextField);
    await tester.enterText(textFieldFinder, '192.168.1.200');
    await tester.pump();

    await tester.tap(find.text('Salvar e Conectar'));
    await tester.pumpAndSettle();

    expect(controller.ollamaBaseUrl.value, 'http://192.168.1.200:11434');
  });
}
