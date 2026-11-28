import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:gpt_markdown/gpt_markdown.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_bubble.dart';

void main() {
  Widget buildTestableWidget(Widget child) {
    return MaterialApp(
      home: Scaffold(
        body: child,
      ),
    );
  }

  group('ChatBubble Widget Tests', () {
    testWidgets('renders user message properly with GptMarkdown', (tester) async {
      final message = ChatMessage(
        text: 'Olá, este é um **texto em negrito** do usuário!',
        isUser: true,
      );

      await tester.pumpWidget(buildTestableWidget(ChatBubble(message: message)));
      await tester.pumpAndSettle();

      expect(find.byType(GptMarkdown), findsOneWidget);
      expect(find.textContaining('Olá, este é um'), findsWidgets);
      // For user message, copy button should not be displayed
      expect(find.byIcon(Icons.copy_rounded), findsNothing);
    });

    testWidgets('renders assistant message with GptMarkdown and copy button', (tester) async {
      final message = ChatMessage(
        text: 'Resposta com `código inline` e **negrito** da IA.',
        isUser: false,
      );

      await tester.pumpWidget(buildTestableWidget(ChatBubble(message: message)));
      await tester.pumpAndSettle();

      expect(find.byType(GptMarkdown), findsOneWidget);
      expect(
        find.byWidgetPredicate(
          (w) => w is RichText && w.text.toPlainText().contains('Resposta com'),
        ),
        findsWidgets,
      );
      // Assistant message should display the copy button
      expect(find.byIcon(Icons.copy_rounded), findsOneWidget);
      expect(find.text('Copiar'), findsOneWidget);

      await tester.tap(find.text('Copiar'));
      await tester.pump();

      expect(find.text('Mensagem copiada para a área de transferência!'), findsOneWidget);
    });

    testWidgets('renders system message as centered chip without GptMarkdown', (tester) async {
      final message = ChatMessage(
        text: 'Sistema: Modelo gemma_2b.litertlm carregado com sucesso!',
        isUser: false,
      );

      await tester.pumpWidget(buildTestableWidget(ChatBubble(message: message)));
      await tester.pumpAndSettle();

      // System message uses custom chip and text, not standard GptMarkdown
      expect(find.byType(GptMarkdown), findsNothing);
      expect(find.byIcon(Icons.info_outline), findsOneWidget);
      expect(find.text('Modelo gemma_2b.litertlm carregado com sucesso!'), findsOneWidget);
      expect(find.byIcon(Icons.copy_rounded), findsNothing);
    });

    testWidgets('renders system error message with error icon and styling', (tester) async {
      final message = ChatMessage(
        text: 'Sistema: Erro ao gerar resposta (timeout)',
        isUser: false,
      );

      await tester.pumpWidget(buildTestableWidget(ChatBubble(message: message)));
      await tester.pumpAndSettle();

      expect(find.byType(GptMarkdown), findsNothing);
      expect(find.byIcon(Icons.error_outline), findsOneWidget);
      expect(find.text('Erro ao gerar resposta (timeout)'), findsOneWidget);
    });

    testWidgets('renders waiting indicator when assistant message text is empty', (tester) async {
      final message = ChatMessage(
        text: '',
        isUser: false,
      );

      await tester.pumpWidget(buildTestableWidget(ChatBubble(message: message)));

      // When message is empty while awaiting streaming tokens, a progress indicator is shown
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
      expect(find.byType(GptMarkdown), findsNothing);
    });

    testWidgets('renders collapsible thinking section when message has thinkingText', (tester) async {
      final message = ChatMessage(
        text: 'A resposta final é 42.',
        thinkingText: 'Analisando o sentido da vida...',
        isUser: false,
      );

      await tester.pumpWidget(buildTestableWidget(ChatBubble(message: message)));
      await tester.pumpAndSettle();

      expect(find.text('Pensamento'), findsOneWidget);
      // Começa recolhido quando a resposta final já está presente
      expect(find.text('Analisando o sentido da vida...'), findsNothing);

      await tester.tap(find.text('Pensamento'));
      await tester.pumpAndSettle();

      expect(find.text('Analisando o sentido da vida...'), findsOneWidget);
    });

    testWidgets('renders Thinking in progress expanded while final text is still empty', (tester) async {
      final message = ChatMessage(
        text: '',
        thinkingText: 'Raciocinando passo a passo...',
        isUser: false,
      );

      await tester.pumpWidget(buildTestableWidget(ChatBubble(message: message)));
      await tester.pump();

      expect(find.text('Pensando...'), findsOneWidget);
      expect(find.text('Raciocinando passo a passo...'), findsOneWidget);
    });
  });
}
