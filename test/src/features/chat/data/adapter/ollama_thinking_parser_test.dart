import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_thinking_parser.dart';

void main() {
  group('OllamaThinkingParser', () {
    test('handles native message.thinking followed by message.content', () {
      final parser = OllamaThinkingParser();

      final c1 = parser.parseJsonChunk({
        'message': {'role': 'assistant', 'thinking': 'Step 1 '},
        'done': false,
      });
      final c2 = parser.parseJsonChunk({
        'message': {'role': 'assistant', 'thinking': 'Step 2'},
        'done': false,
      });
      final c3 = parser.parseJsonChunk({
        'message': {'role': 'assistant', 'content': 'Final answer'},
        'done': true,
      });

      expect(c1.thinkingDelta, 'Step 1 ');
      expect(c1.contentDelta, '');
      expect(c1.isThinking, isTrue);
      expect(c1.isDone, isFalse);

      expect(c2.thinkingDelta, 'Step 2');
      expect(c2.contentDelta, '');
      expect(c2.isThinking, isTrue);

      expect(c3.thinkingDelta, '');
      expect(c3.contentDelta, 'Final answer');
      expect(c3.isThinking, isFalse);
      expect(c3.isDone, isTrue);
    });

    test('handles top-level thinking and response fields (/api/generate)', () {
      final parser = OllamaThinkingParser();

      final c1 = parser.parseJsonChunk({
        'thinking': 'Reasoning',
        'response': '',
        'done': false,
      });
      final c2 = parser.parseJsonChunk({
        'thinking': '',
        'response': 'Done!',
        'done': true,
      });

      expect(c1.thinkingDelta, 'Reasoning');
      expect(c1.isThinking, isTrue);

      expect(c2.contentDelta, 'Done!');
      expect(c2.isThinking, isFalse);
      expect(c2.isDone, isTrue);
    });

    test('extracts <think>...</think> split across arbitrary chunks', () {
      final parser = OllamaThinkingParser();
      final inputs = [
        '<th',
        'ink>Anal',
        'yzing ',
        'options</t',
        'hink>',
        'Answer',
        ' text',
      ];

      final chunks = [
        for (var i = 0; i < inputs.length; i++)
          parser.feed(
            rawContent: inputs[i],
            isDone: i == inputs.length - 1,
          ),
      ];

      expect(chunks.map((c) => c.thinkingDelta).join(), 'Analyzing options');
      expect(chunks.map((c) => c.contentDelta).join(), 'Answer text');
      expect(chunks.any((c) => c.isThinking), isTrue);
      expect(chunks.last.isThinking, isFalse);
      expect(chunks.last.isDone, isTrue);
    });

    test('strips whitespace before <think> tag and keeps post-</think> content', () {
      final parser = OllamaThinkingParser();

      final c1 = parser.feed(rawContent: '\n  <think>Plan</think>Hello');
      final c2 = parser.feed(rawContent: ' world', isDone: true);

      expect(c1.thinkingDelta, 'Plan');
      expect(c1.contentDelta, 'Hello');
      expect(c1.isThinking, isFalse);

      expect(c2.contentDelta, ' world');
      expect(c2.isDone, isTrue);
    });

    test('handles plain text containing angle brackets without false <think>', () {
      final parser = OllamaThinkingParser();

      final c1 = parser.feed(rawContent: '2 < 3 ');
      final c2 = parser.feed(rawContent: 'and 4 > 1', isDone: true);

      expect(c1.thinkingDelta, '');
      expect(c1.contentDelta, '2 < 3 ');
      expect(c1.isThinking, isFalse);

      expect(c2.thinkingDelta, '');
      expect(c2.contentDelta, 'and 4 > 1');
      expect(c2.isDone, isTrue);
    });

    test('handles empty <think></think> with no reasoning emitted', () {
      final parser = OllamaThinkingParser();

      final c1 = parser.feed(rawContent: '<think></think>Direct response', isDone: true);

      expect(c1.thinkingDelta, '');
      expect(c1.contentDelta, 'Direct response');
      expect(c1.isThinking, isFalse);
      expect(c1.isDone, isTrue);
    });

    test('flushes buffered partial </think tail inside unclosed think block on done', () {
      final parser = OllamaThinkingParser();

      final c1 = parser.feed(rawContent: '<think>Partial thought</th');
      final c2 = parser.feed(rawContent: '', isDone: true);

      expect(c1.thinkingDelta, 'Partial thought');
      expect(c1.isThinking, isTrue);

      expect(c2.thinkingDelta, '</th');
      expect(c2.isThinking, isFalse);
      expect(c2.isDone, isTrue);
    });
  });
}
