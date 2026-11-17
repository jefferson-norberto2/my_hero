import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/services/file_export_service.dart';

void main() {
  group('FileExportService - Content Formatting & Sanitization', () {
    late FileExportService service;

    setUp(() {
      service = FileExportService(
        saveFileDialog: ({dialogTitle, fileName, bytes}) async => null,
        isAndroid: () => false,
      );
    });

    test('single assistant message omits thinkingContent and author headings', () {
      final message = ChatMessage(
        content: 'Resposta final clara.',
        thinkingContent: 'Raciocínio interno que deve ser omitido.',
        isUser: false,
      );

      final payload = service.buildSingleMessagePayload(message: message);

      expect(payload.formattedContent, 'Resposta final clara.');
      expect(payload.formattedContent, isNot(contains('Raciocínio')));
      expect(payload.formattedContent, isNot(contains('## Assistente')));
      expect(payload.suggestedFileName, 'resposta.md');
    });

    test('single user message exports clean content with mensagem.md suggestion', () {
      final message = ChatMessage(
        content: '```markdown\n# Mantido no usuário\n```',
        isUser: true,
      );

      final payload = service.buildSingleMessagePayload(message: message);

      expect(payload.formattedContent, '```markdown\n# Mantido no usuário\n```');
      expect(payload.suggestedFileName, 'mensagem.md');
    });

    test('strips outer ```markdown, ```md, and ``` fences on single assistant message', () {
      const fencedMarkdown = '```markdown\n# Título\n\nParágrafo.\n```';
      const fencedMd = '```md\n## Subtítulo\n\nTexto.\n```';
      const fencedPlain = '```\nLinha 1\nLinha 2\n```';

      expect(
        service
            .buildSingleMessagePayload(
              message: ChatMessage(content: fencedMarkdown, isUser: false),
            )
            .formattedContent,
        '# Título\n\nParágrafo.',
      );
      expect(
        service
            .buildSingleMessagePayload(
              message: ChatMessage(content: fencedMd, isUser: false),
            )
            .formattedContent,
        '## Subtítulo\n\nTexto.',
      );
      expect(
        service
            .buildSingleMessagePayload(
              message: ChatMessage(content: fencedPlain, isUser: false),
            )
            .formattedContent,
        'Linha 1\nLinha 2',
      );
    });

    test('preserves nested code blocks inside a single outer ```markdown wrapper', () {
      const content = '''```markdown
# Guia Rápido

```python
print("olá")
```

Fim do guia.
```''';

      final result = service.stripOuterMarkdownFence(content);

      expect(result, '''# Guia Rápido

```python
print("olá")
```

Fim do guia.''');
    });

    test('does not strip multiple separate code blocks or language-specific blocks', () {
      const twoBlocks = '''```
Bloco 1
```

Texto intermediário

```
Bloco 2
```''';
      const pythonBlock = '''```python
def soma(a, b):
    return a + b
```''';

      expect(service.stripOuterMarkdownFence(twoBlocks), twoBlocks);
      expect(service.stripOuterMarkdownFence(pythonBlock), pythonBlock);
    });

    test('derives filename preserving extension from immediately preceding user document attachment', () {
      final mdPrompt = ChatMessage(
        content: 'Revise este artigo',
        isUser: true,
        attachment: ChatAttachment(
          name: 'artigo.md',
          type: AttachmentType.document,
          bytes: Uint8List(0),
        ),
      );
      final pyPrompt = ChatMessage(
        content: 'Otimize este script',
        isUser: true,
        attachment: ChatAttachment(
          name: 'script.py',
          type: AttachmentType.document,
          bytes: Uint8List(0),
        ),
      );
      final pdfPrompt = ChatMessage(
        content: 'Resuma o contrato',
        isUser: true,
        attachment: ChatAttachment(
          name: 'contrato.pdf',
          type: AttachmentType.document,
          bytes: Uint8List(0),
        ),
      );
      final imgPrompt = ChatMessage(
        content: 'Descreva a foto',
        isUser: true,
        attachment: ChatAttachment(
          name: 'foto.png',
          type: AttachmentType.image,
          bytes: Uint8List(0),
        ),
      );
      final reply = ChatMessage(content: 'Ok', isUser: false);

      expect(
        service
            .buildSingleMessagePayload(message: reply, previousMessage: mdPrompt)
            .suggestedFileName,
        'artigo_resposta.md',
      );
      expect(
        service
            .buildSingleMessagePayload(message: reply, previousMessage: pyPrompt)
            .suggestedFileName,
        'script_resposta.py',
      );
      expect(
        service
            .buildSingleMessagePayload(message: reply, previousMessage: pdfPrompt)
            .suggestedFileName,
        'contrato_resposta.md',
      );
      expect(
        service
            .buildSingleMessagePayload(message: reply, previousMessage: imgPrompt)
            .suggestedFileName,
        'resposta.md',
      );
    });

    test('buildSelectedMessagesPayload finds preceding user attachment skipping system banners', () {
      final messages = [
        ChatMessage(
          content: 'Revise',
          isUser: true,
          attachment: ChatAttachment(
            name: 'script.py',
            type: AttachmentType.document,
            bytes: Uint8List(0),
          ),
        ),
        ChatMessage(content: 'Sistema: Estilo alterado', isUser: false),
        ChatMessage(
          content: 'print("ok")',
          thinkingContent: 'Pensando no print',
          isUser: false,
        ),
      ];

      final payload = service.buildSelectedMessagesPayload(
        conversationTitle: 'Nova Conversa',
        allMessages: messages,
        selectedIndices: [2],
      );

      expect(payload, isNotNull);
      expect(payload!.suggestedFileName, 'script_resposta.py');
      expect(payload.formattedContent, 'print("ok")');
    });

    test('multi-message selection sorts chronologically regardless of click order and formats Markdown', () {
      final messages = [
        ChatMessage(content: 'Sistema: Modelo carregado', isUser: false),
        ChatMessage(content: 'Primeira pergunta', isUser: true),
        ChatMessage(
          content: 'Primeira resposta',
          thinkingContent: 'Raciocínio 1',
          isUser: false,
        ),
        ChatMessage(content: 'Segunda pergunta', isUser: true),
        ChatMessage(
          content: 'Segunda resposta',
          thinkingContent: 'Raciocínio 2',
          isUser: false,
        ),
      ];

      final payload = service.buildSelectedMessagesPayload(
        conversationTitle: 'Nova Conversa',
        allMessages: messages,
        selectedIndices: [4, 1, 3],
      );

      expect(payload, isNotNull);
      expect(payload!.suggestedFileName, 'nova_conversa.md');
      expect(
        payload.formattedContent,
        '# Nova Conversa\n\n'
        '## Usuário\n\n'
        'Primeira pergunta\n\n'
        '## Usuário\n\n'
        'Segunda pergunta\n\n'
        '## Assistente\n\n'
        'Segunda resposta',
      );
      expect(payload.formattedContent, isNot(contains('Raciocínio')));
      expect(payload.formattedContent, isNot(contains('Sistema:')));
    });

    test('sanitizes conversation titles with accents, punctuation, and reserved names', () {
      expect(
        service.sanitizeConversationFileName('Nova Conversa'),
        'nova_conversa.md',
      );
      expect(
        service.sanitizeConversationFileName('Análise: Código & Dados?'),
        'analise_codigo_dados.md',
      );
      expect(
        service.sanitizeConversationFileName('CON'),
        'con_file.md',
      );
      expect(
        service.sanitizeConversationFileName('???...'),
        'conversa.md',
      );
    });
  });

  group('FileExportService - Cross-Platform File Writing', () {
    test('desktop (Linux/Windows) passes UTF-8 bytes to saveFile and explicitly writes UTF-8 string to disk', () async {
      final tempDir = await Directory.systemTemp.createTemp('file_export_test_');
      addTearDown(() async {
        if (await tempDir.exists()) {
          await tempDir.delete(recursive: true);
        }
      });

      final targetFile = File('${tempDir.path}/nova_conversa.md');
      String? capturedTitle;
      String? capturedFileName;
      Uint8List? capturedBytes;

      final service = FileExportService(
        isAndroid: () => false,
        saveFileDialog: ({dialogTitle, fileName, bytes}) async {
          capturedTitle = dialogTitle;
          capturedFileName = fileName;
          capturedBytes = bytes;
          return targetFile.path;
        },
      );

      final now = DateTime(2026, 3, 1);
      final conversation = Conversation(
        id: 'conv-1',
        title: 'Nova Conversa',
        messages: [
          ChatMessage(content: 'Olá! Como usar acentuação: çãõ?', isUser: true),
          ChatMessage(
            content: 'Use UTF-8 sempre.',
            thinkingContent: 'Raciocínio secreto',
            isUser: false,
          ),
        ],
        createdAt: now,
        updatedAt: now,
      );

      final saved = await service.exportConversation(conversation);

      expect(saved, isTrue);
      expect(capturedTitle, 'Exportar conversa');
      expect(capturedFileName, 'nova_conversa.md');
      expect(capturedBytes, isNotNull);

      final expectedContent = '# Nova Conversa\n\n'
          '## Usuário\n\n'
          'Olá! Como usar acentuação: çãõ?\n\n'
          '## Assistente\n\n'
          'Use UTF-8 sempre.';
      expect(utf8.decode(capturedBytes!), expectedContent);
      expect(await targetFile.exists(), isTrue);
      expect(await targetFile.readAsString(encoding: utf8), expectedContent);
    });

    test('Android relies on SAF saveFile bytes and does not perform File.writeAsString', () async {
      Uint8List? capturedBytes;
      String? capturedFileName;

      final service = FileExportService(
        isAndroid: () => true,
        saveFileDialog: ({dialogTitle, fileName, bytes}) async {
          capturedFileName = fileName;
          capturedBytes = bytes;
          return 'content://com.android.providers.downloads/document/42';
        },
      );

      final saved = await service.exportSelectedMessages(
        conversationTitle: 'Teste Android',
        allMessages: [
          ChatMessage(content: 'Conteúdo exportado', isUser: false),
        ],
        selectedIndices: [0],
      );

      expect(saved, isTrue);
      expect(capturedFileName, 'resposta.md');
      expect(utf8.decode(capturedBytes!), 'Conteúdo exportado');
    });

    test('returns false without throwing when user cancels the save dialog', () async {
      final service = FileExportService(
        isAndroid: () => false,
        saveFileDialog: ({dialogTitle, fileName, bytes}) async => null,
      );

      final saved = await service.exportSelectedMessages(
        conversationTitle: 'Teste Cancelado',
        allMessages: [
          ChatMessage(content: 'Olá', isUser: true),
        ],
        selectedIndices: [0],
      );

      expect(saved, isFalse);
    });
  });
}
