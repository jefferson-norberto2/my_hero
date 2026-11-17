import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:gpt_markdown/gpt_markdown.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_bubble.dart';
import 'package:my_hero/src/features/chat/ui/widgets/thinking_block_widget.dart';

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
      expect(find.byType(ThinkingBlockWidget), findsNothing);
      expect(find.byType(GptMarkdown), findsNothing);
    });

    testWidgets('shows Pensando block during thinking and lets user expand live content', (tester) async {
      final message = ChatMessage(
        content: '',
        thinkingContent: 'Avaliando possibilidades...',
        isThinking: true,
        isUser: false,
      );

      await tester.pumpWidget(buildTestableWidget(ChatBubble(message: message)));
      await tester.pump();

      expect(find.byType(ThinkingBlockWidget), findsOneWidget);
      expect(find.text('Pensando...'), findsOneWidget);
      expect(find.byType(GptMarkdown), findsNothing);
      expect(find.text('Avaliando possibilidades...'), findsNothing);

      // User can expand even while thinking is still streaming
      await tester.tap(find.text('Pensando...'));
      await tester.pump();

      expect(find.text('Pensando...'), findsOneWidget);
      expect(find.text('Avaliando possibilidades...'), findsOneWidget);
    });

    testWidgets('defaults collapsed after thinking finishes and toggles Ver/Ocultar raciocínio', (tester) async {
      final message = ChatMessage(
        content: 'Resposta final concluída.',
        thinkingContent: 'Linha 1 do raciocínio\nLinha 2 do raciocínio',
        isThinking: false,
        isUser: false,
      );

      await tester.pumpWidget(buildTestableWidget(ChatBubble(message: message)));
      await tester.pumpAndSettle();

      expect(find.byType(ThinkingBlockWidget), findsOneWidget);
      expect(find.text('Ver raciocínio'), findsOneWidget);
      expect(find.text('Ocultar raciocínio'), findsNothing);
      expect(find.textContaining('Linha 1 do raciocínio'), findsNothing);
      expect(find.byType(GptMarkdown), findsOneWidget);

      await tester.tap(find.text('Ver raciocínio'));
      await tester.pumpAndSettle();

      expect(find.text('Ocultar raciocínio'), findsOneWidget);
      expect(find.textContaining('Linha 1 do raciocínio'), findsOneWidget);

      await tester.tap(find.text('Ocultar raciocínio'));
      await tester.pumpAndSettle();

      expect(find.text('Ver raciocínio'), findsOneWidget);
      expect(find.textContaining('Linha 1 do raciocínio'), findsNothing);
    });

    testWidgets('does not render ThinkingBlockWidget when message has no reasoning', (tester) async {
      final message = ChatMessage(
        content: 'Resposta direta sem raciocínio.',
        thinkingContent: '',
        isThinking: false,
        isUser: false,
      );

      await tester.pumpWidget(buildTestableWidget(ChatBubble(message: message)));
      await tester.pumpAndSettle();

      expect(find.byType(ThinkingBlockWidget), findsNothing);
      expect(find.byType(GptMarkdown), findsOneWidget);
    });

    testWidgets('triggers onLongPress on user and assistant bubbles outside selection mode', (tester) async {
      var userLongPressed = 0;
      var assistantLongPressed = 0;

      await tester.pumpWidget(
        buildTestableWidget(
          Column(
            children: [
              ChatBubble(
                message: ChatMessage(content: 'Pergunta do usuário', isUser: true),
                onLongPress: () => userLongPressed++,
              ),
              ChatBubble(
                message: ChatMessage(content: 'Resposta do assistente', isUser: false),
                onLongPress: () => assistantLongPressed++,
              ),
            ],
          ),
        ),
      );
      await tester.pumpAndSettle();

      await tester.longPress(find.text('Pergunta do usuário'));
      await tester.longPress(find.text('Resposta do assistente'));

      expect(userLongPressed, 1);
      expect(assistantLongPressed, 1);
    });

    testWidgets('renders Checkbox and toggles selection on tap anywhere on row while omitting Checkbox for system banners', (tester) async {
      var userToggled = 0;
      var assistantToggled = 0;

      await tester.pumpWidget(
        buildTestableWidget(
          Column(
            children: [
              ChatBubble(
                message: ChatMessage(content: 'Sistema: Modelo carregado', isUser: false),
                isSelectionMode: true,
                isSelectable: false,
                onToggleSelection: () {},
              ),
              ChatBubble(
                message: ChatMessage(content: 'Oi', isUser: true),
                isSelectionMode: true,
                isSelected: true,
                onToggleSelection: () => userToggled++,
              ),
              ChatBubble(
                message: ChatMessage(content: 'Resposta longa aqui', isUser: false),
                isSelectionMode: true,
                isSelected: false,
                onToggleSelection: () => assistantToggled++,
              ),
            ],
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.byType(Checkbox), findsNWidgets(2));

      // Tapping Checkbox toggles once.
      await tester.tap(find.byType(Checkbox).first);
      // Tapping bubble text/copy area in selection mode toggles selection instead of copying.
      await tester.tap(find.text('Copiar'));
      await tester.pump();

      expect(userToggled, 1);
      expect(assistantToggled, 1);
      expect(find.text('Mensagem copiada para a área de transferência!'), findsNothing);
    });
  });
}
