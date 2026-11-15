import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_metrics.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_memory_parser.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_metrics_badge.dart';

void main() {
  Widget buildSubject(ChatMetrics metrics) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          actions: [
            ChatMetricsBadge(
              metrics: metrics,
              onTap: () {},
            ),
          ],
        ),
      ),
    );
  }

  group('ChatMetricsBadge Widget Tests', () {
    testWidgets('renders memory and GPU tag with VRAM/RAM tooltip', (tester) async {
      final metrics = ChatMetrics(
        contextLimit: 8192,
        memoryAllocation: OllamaMemoryParser.fromBytes(
          sizeBytes: 4080218931,
          vramBytes: 4080218931,
        ),
      );

      await tester.pumpWidget(buildSubject(metrics));

      expect(find.text('3.8 GB'), findsOneWidget);
      expect(find.text('GPU'), findsOneWidget);
      expect(find.byTooltip('VRAM: 3.8 GB | RAM: 0.0 GB'), findsOneWidget);
    });

    testWidgets('renders memory and CPU tag with RAM tooltip', (tester) async {
      final metrics = ChatMetrics(
        contextLimit: 8192,
        memoryAllocation: OllamaMemoryParser.fromBytes(
          sizeBytes: 5798205849,
          vramBytes: 0,
        ),
      );

      await tester.pumpWidget(buildSubject(metrics));

      expect(find.text('5.4 GB'), findsOneWidget);
      expect(find.text('CPU'), findsOneWidget);
      expect(find.byTooltip('Memória RAM (CPU): 5.4 GB'), findsOneWidget);
    });

    testWidgets('renders total memory and GPU/CPU tag in hybrid mode', (tester) async {
      final metrics = ChatMetrics(
        contextLimit: 8192,
        memoryAllocation: OllamaMemoryParser.fromBytes(
          sizeBytes: 5798205849,
          vramBytes: 3221225472,
        ),
      );

      await tester.pumpWidget(buildSubject(metrics));

      expect(find.text('5.4 GB'), findsOneWidget);
      expect(find.text('GPU/CPU'), findsOneWidget);
      expect(find.byTooltip('VRAM: 3.0 GB | RAM: 2.4 GB'), findsOneWidget);
    });

    testWidgets('renders 0.0 GB without active hardware tag when unloaded', (tester) async {
      const metrics = ChatMetrics(contextLimit: 8192);

      await tester.pumpWidget(buildSubject(metrics));

      expect(find.text('0.0 GB'), findsOneWidget);
      expect(find.text('GPU'), findsNothing);
      expect(find.text('CPU'), findsNothing);
      expect(find.text('GPU/CPU'), findsNothing);
    });

    testWidgets('displays active Ctx percentage and active/max context tooltip', (tester) async {
      const metrics = ChatMetrics(
        promptTokens: 2048,
        contextLimit: 8192,
        maxContextLength: 32768,
      );

      await tester.pumpWidget(buildSubject(metrics));

      expect(find.text('Ctx: 25% (restam 75%)'), findsOneWidget);
      expect(
        find.byTooltip('Contexto ativo: 8.192 / Máx. do modelo: 32.768 tokens'),
        findsOneWidget,
      );
    });
  });
}
