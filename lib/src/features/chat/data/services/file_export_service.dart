import 'dart:convert';
import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/shared/models/chat_export_payload.dart';

/// Platform save-file dialog callback matching [FilePicker.saveFile].
typedef SaveFileDialogCallback = Future<String?> Function({
  String? dialogTitle,
  String? fileName,
  Uint8List? bytes,
});

/// Formats, sanitizes, and exports chat conversations and messages to local
/// files across Linux, Windows, and Android.
class FileExportService {
  static const String exportDialogTitle = 'Exportar conversa';
  static const String defaultConversationTitle = 'Conversa sem título';
  static const String defaultConversationFileStem = 'conversa';
  static const String defaultAssistantFileName = 'resposta.md';
  static const String defaultUserFileName = 'mensagem.md';

  static const int _maxFileStemLength = 64;

  static const Set<String> _windowsReservedNames = <String>{
    'con',
    'prn',
    'aux',
    'nul',
    'com1',
    'com2',
    'com3',
    'com4',
    'com5',
    'com6',
    'com7',
    'com8',
    'com9',
    'lpt1',
    'lpt2',
    'lpt3',
    'lpt4',
    'lpt5',
    'lpt6',
    'lpt7',
    'lpt8',
    'lpt9',
  };

  static const Set<String> _nonTextExtensions = <String>{
    'pdf',
    'doc',
    'docx',
    'xls',
    'xlsx',
    'ppt',
    'pptx',
    'zip',
    'rar',
    '7z',
    'tar',
    'gz',
    'png',
    'jpg',
    'jpeg',
    'gif',
    'webp',
    'bmp',
    'ico',
    'mp3',
    'wav',
    'ogg',
    'm4a',
    'flac',
    'aac',
    'mp4',
    'avi',
    'mkv',
    'mov',
    'exe',
    'dll',
    'so',
    'bin',
  };

  static const Map<String, String> _diacriticMap = <String, String>{
    'à': 'a',
    'á': 'a',
    'â': 'a',
    'ã': 'a',
    'ä': 'a',
    'å': 'a',
    'À': 'A',
    'Á': 'A',
    'Â': 'A',
    'Ã': 'A',
    'Ä': 'A',
    'Å': 'A',
    'è': 'e',
    'é': 'e',
    'ê': 'e',
    'ë': 'e',
    'È': 'E',
    'É': 'E',
    'Ê': 'E',
    'Ë': 'E',
    'ì': 'i',
    'í': 'i',
    'î': 'i',
    'ï': 'i',
    'Ì': 'I',
    'Í': 'I',
    'Î': 'I',
    'Ï': 'I',
    'ò': 'o',
    'ó': 'o',
    'ô': 'o',
    'õ': 'o',
    'ö': 'o',
    'Ò': 'O',
    'Ó': 'O',
    'Ô': 'O',
    'Õ': 'O',
    'Ö': 'O',
    'ù': 'u',
    'ú': 'u',
    'û': 'u',
    'ü': 'u',
    'Ù': 'U',
    'Ú': 'U',
    'Û': 'U',
    'Ü': 'U',
    'ç': 'c',
    'Ç': 'C',
    'ñ': 'n',
    'Ñ': 'N',
    'ý': 'y',
    'ÿ': 'y',
    'Ý': 'Y',
  };

  static final RegExp _openingFencePattern = RegExp(
    r'^(`{3,})(?:markdown|md)?$',
    caseSensitive: false,
  );
  static final RegExp _fenceLinePattern = RegExp(r'^(`{3,})(.*)$');

  final SaveFileDialogCallback _saveFileDialog;
  final bool Function() _isAndroid;

  FileExportService({
    SaveFileDialogCallback? saveFileDialog,
    bool Function()? isAndroid,
  })  : _saveFileDialog = saveFileDialog ?? _defaultSaveFileDialog,
        _isAndroid = isAndroid ?? _defaultIsAndroid;

  static Future<String?> _defaultSaveFileDialog({
    String? dialogTitle,
    String? fileName,
    Uint8List? bytes,
  }) {
    return FilePicker.saveFile(
      dialogTitle: dialogTitle,
      fileName: fileName,
      bytes: bytes,
    );
  }

  static bool _defaultIsAndroid() => !kIsWeb && Platform.isAndroid;

  /// Exports a full [conversation] as a structured Markdown file.
  Future<bool> exportConversation(Conversation conversation) {
    return exportConversationMessages(
      title: conversation.title,
      messages: conversation.messages,
    );
  }

  /// Exports all exportable [messages] under [title] as a structured Markdown
  /// document.
  Future<bool> exportConversationMessages({
    required String title,
    required List<ChatMessage> messages,
  }) async {
    final exportable = messages.where((m) => m.isExportable).toList();
    if (exportable.isEmpty) {
      return false;
    }

    final payload = buildConversationPayload(
      title: title,
      messages: exportable,
    );
    return saveExportPayload(payload);
  }

  /// Exports the messages at [selectedIndices] from [allMessages] in
  /// chronological order.
  Future<bool> exportSelectedMessages({
    required String conversationTitle,
    required List<ChatMessage> allMessages,
    required Iterable<int> selectedIndices,
  }) async {
    final payload = buildSelectedMessagesPayload(
      conversationTitle: conversationTitle,
      allMessages: allMessages,
      selectedIndices: selectedIndices,
    );
    if (payload == null) {
      return false;
    }

    return saveExportPayload(payload);
  }

  /// Prompts the user to pick a save destination and writes [payload] in UTF-8.
  ///
  /// On Android, [FilePicker.saveFile] persists [bytes] directly through SAF.
  /// On Linux and Windows, [FilePicker.saveFile] returns the destination path
  /// and this method explicitly writes the UTF-8 content to disk.
  /// Returns `false` without throwing when the user cancels the dialog.
  Future<bool> saveExportPayload(ChatExportPayload payload) async {
    final bytes = Uint8List.fromList(utf8.encode(payload.formattedContent));
    final savedPath = await _saveFileDialog(
      dialogTitle: exportDialogTitle,
      fileName: payload.suggestedFileName,
      bytes: bytes,
    );

    if (savedPath == null || savedPath.trim().isEmpty) {
      return false;
    }

    if (!_isAndroid()) {
      await File(savedPath).writeAsString(
        payload.formattedContent,
        encoding: utf8,
      );
    }

    return true;
  }

  /// Prepares an export payload for [selectedIndices] sorted chronologically.
  ChatExportPayload? buildSelectedMessagesPayload({
    required String conversationTitle,
    required List<ChatMessage> allMessages,
    required Iterable<int> selectedIndices,
  }) {
    final sortedIndices = selectedIndices
        .where(
          (i) =>
              i >= 0 &&
              i < allMessages.length &&
              !allMessages[i].isSystemMessage,
        )
        .toSet()
        .toList()
      ..sort();

    if (sortedIndices.isEmpty) {
      return null;
    }

    if (sortedIndices.length == 1) {
      final selectedIndex = sortedIndices.first;
      return buildSingleMessagePayload(
        message: allMessages[selectedIndex],
        previousMessage: _findImmediatelyPrecedingUserMessage(
          allMessages,
          selectedIndex,
        ),
      );
    }

    final selectedMessages = <ChatMessage>[
      for (final index in sortedIndices) allMessages[index],
    ];
    return buildConversationPayload(
      title: conversationTitle,
      messages: selectedMessages,
    );
  }

  /// Builds the export payload for a single selected [message], omitting
  /// reasoning, stripping outer Markdown fences on assistant replies, and
  /// deriving the file name from [previousMessage]'s attachment when present.
  ChatExportPayload buildSingleMessagePayload({
    required ChatMessage message,
    ChatMessage? previousMessage,
  }) {
    final rawContent = message.content.trim();
    final formattedContent = message.isUser
        ? rawContent
        : stripOuterMarkdownFence(rawContent);

    return ChatExportPayload(
      formattedContent: formattedContent,
      suggestedFileName: suggestSingleMessageFileName(
        message: message,
        previousMessage: previousMessage,
      ),
    );
  }

  /// Builds a structured Markdown document for multiple messages or a full
  /// conversation, omitting all `thinkingContent` and internal system banners.
  ChatExportPayload buildConversationPayload({
    required String title,
    required List<ChatMessage> messages,
  }) {
    final effectiveTitle =
        title.trim().isNotEmpty ? title.trim() : defaultConversationTitle;

    final buffer = StringBuffer()..writeln('# $effectiveTitle');

    for (final message in messages) {
      if (message.isSystemMessage) continue;
      final cleanContent = message.content.trim();
      if (cleanContent.isEmpty) continue;

      final authorHeading = message.isUser ? '## Usuário' : '## Assistente';
      buffer
        ..writeln()
        ..writeln(authorHeading)
        ..writeln()
        ..writeln(cleanContent);
    }

    return ChatExportPayload(
      formattedContent: buffer.toString().trimRight(),
      suggestedFileName: sanitizeConversationFileName(effectiveTitle),
    );
  }

  /// Removes an outer Markdown code-fence wrapper (` ```markdown `, ` ```md `,
  /// or ` ``` `) when it wraps the entire response as a single outer block.
  String stripOuterMarkdownFence(String content) {
    final trimmed = content.trim();
    if (trimmed.isEmpty) return '';

    final normalized =
        trimmed.replaceAll('\r\n', '\n').replaceAll('\r', '\n');
    final lines = normalized.split('\n');
    if (lines.length < 2) return trimmed;

    final firstLine = lines.first.trim();
    final openingMatch = _openingFencePattern.firstMatch(firstLine);
    if (openingMatch == null) return trimmed;

    final fenceLength = openingMatch.group(1)!.length;
    final lastLine = lines.last.trim();
    if (!RegExp('^`{$fenceLength}\$').hasMatch(lastLine)) {
      return trimmed;
    }

    // Track fence nesting so two separate blocks (e.g. ``` A ``` ... ``` B ```)
    // are never falsely stripped, while nested blocks inside ```markdown work.
    var depth = 1;
    for (var i = 1; i < lines.length - 1; i++) {
      final line = lines[i].trim();
      final match = _fenceLinePattern.firstMatch(line);
      if (match == null) continue;

      final tickCount = match.group(1)!.length;
      final infoString = match.group(2)!.trim();

      if (infoString.isNotEmpty) {
        if (fenceLength == 3 || tickCount >= fenceLength) {
          depth++;
        }
      } else {
        if (depth > 1) {
          depth--;
        } else if (tickCount >= fenceLength) {
          return trimmed;
        }
      }
    }

    if (depth != 1) return trimmed;

    final innerLines = lines.sublist(1, lines.length - 1);
    var start = 0;
    var end = innerLines.length;
    while (start < end && innerLines[start].trim().isEmpty) {
      start++;
    }
    while (end > start && innerLines[end - 1].trim().isEmpty) {
      end--;
    }
    if (start >= end) return '';

    return innerLines.sublist(start, end).join('\n');
  }

  /// Suggests a sanitized file name for exporting a single [message].
  String suggestSingleMessageFileName({
    required ChatMessage message,
    ChatMessage? previousMessage,
  }) {
    if (message.isUser) {
      return defaultUserFileName;
    }

    final attachment =
        previousMessage != null && previousMessage.isUser
            ? previousMessage.attachment
            : null;
    if (attachment == null) {
      return defaultAssistantFileName;
    }

    return deriveAttachmentResponseFileName(attachment);
  }

  /// Derives a response file name from [attachment] preserving its original
  /// text/code extension (e.g. `artigo.md` -> `artigo_resposta.md`).
  String deriveAttachmentResponseFileName(ChatAttachment attachment) {
    if (attachment.type != AttachmentType.document) {
      return defaultAssistantFileName;
    }

    final baseName = _extractBaseName(attachment.name);
    if (baseName.isEmpty) {
      return defaultAssistantFileName;
    }

    final dotIndex = baseName.lastIndexOf('.');
    if (dotIndex > 0 && dotIndex < baseName.length - 1) {
      final rawStem = baseName.substring(0, dotIndex);
      final rawExt = baseName.substring(dotIndex + 1);
      final safeStem = _sanitizeFileStem(rawStem, lowercase: false);
      final cleanExt = rawExt.replaceAll(RegExp(r'[^a-zA-Z0-9]'), '');

      if (safeStem.isEmpty) {
        return defaultAssistantFileName;
      }

      final targetExt =
          cleanExt.isEmpty || _nonTextExtensions.contains(cleanExt.toLowerCase())
              ? 'md'
              : cleanExt;
      return '${safeStem}_resposta.$targetExt';
    }

    final safeStem = _sanitizeFileStem(baseName, lowercase: false);
    if (safeStem.isEmpty) {
      return defaultAssistantFileName;
    }
    return '${safeStem}_resposta.md';
  }

  /// Sanitizes a conversation [title] into a cross-platform `.md` file name.
  String sanitizeConversationFileName(String title) {
    var safeStem = _sanitizeFileStem(title, lowercase: true);
    if (safeStem.isEmpty) {
      safeStem = defaultConversationFileStem;
    }
    return '$safeStem.md';
  }

  ChatMessage? _findImmediatelyPrecedingUserMessage(
    List<ChatMessage> messages,
    int index,
  ) {
    for (var i = index - 1; i >= 0; i--) {
      final candidate = messages[i];
      if (candidate.isSystemMessage) continue;
      return candidate.isUser ? candidate : null;
    }
    return null;
  }

  String _extractBaseName(String rawPath) {
    final trimmed = rawPath.trim();
    if (trimmed.isEmpty) return '';
    final segments = trimmed.split(RegExp(r'[/\\]+'));
    for (var i = segments.length - 1; i >= 0; i--) {
      if (segments[i].trim().isNotEmpty) {
        return segments[i].trim();
      }
    }
    return '';
  }

  String _sanitizeFileStem(String raw, {required bool lowercase}) {
    final buffer = StringBuffer();
    for (final rune in raw.trim().runes) {
      final ch = String.fromCharCode(rune);
      buffer.write(_diacriticMap[ch] ?? ch);
    }

    var sanitized = buffer.toString();
    if (lowercase) {
      sanitized = sanitized.toLowerCase();
    }

    sanitized = sanitized
        .replaceAll(RegExp(r'[^a-zA-Z0-9_-]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .replaceAll(RegExp(r'^[_-]+|[_-]+$'), '');

    if (sanitized.length > _maxFileStemLength) {
      sanitized = sanitized
          .substring(0, _maxFileStemLength)
          .replaceAll(RegExp(r'[_-]+$'), '');
    }

    if (sanitized.isNotEmpty &&
        _windowsReservedNames.contains(sanitized.toLowerCase())) {
      sanitized = '${sanitized}_file';
    }

    return sanitized;
  }
}
