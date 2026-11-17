import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gpt_markdown/gpt_markdown.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/ui/widgets/thinking_block_widget.dart';
import 'package:url_launcher/url_launcher.dart';

/// Displays an individual user, assistant, or system message in the chat list.
class ChatBubble extends StatelessWidget {
  final ChatMessage message;
  final bool isSelectionMode;
  final bool isSelected;
  final bool isSelectable;
  final VoidCallback? onLongPress;
  final VoidCallback? onToggleSelection;

  const ChatBubble({
    super.key,
    required this.message,
    this.isSelectionMode = false,
    this.isSelected = false,
    this.isSelectable = true,
    this.onLongPress,
    this.onToggleSelection,
  });

  bool get _isSystemMessage => message.isSystemMessage;

  @override
  Widget build(BuildContext context) {
    if (_isSystemMessage) {
      return _buildSystemMessage(context);
    }

    final isUser = message.isUser;
    final theme = Theme.of(context);
    final appColors = AppThemeColors.of(context);
    final maxWidth = MediaQuery.sizeOf(context).width * 0.85;
    final hasThinking = !isUser && message.hasThinking;
    final isWaitingFirstToken =
        !isUser && message.content.isEmpty && !hasThinking;

    final borderColor = isSelected
        ? theme.colorScheme.primary
        : (isUser
            ? appColors.userBubbleBorder
            : appColors.assistantBubbleBorder);

    final bubbleContainer = Container(
      constraints: BoxConstraints(maxWidth: maxWidth),
      margin: const EdgeInsets.symmetric(vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: isUser
            ? appColors.userBubbleBackground
            : appColors.assistantBubbleBackground,
        borderRadius: BorderRadius.only(
          topLeft: const Radius.circular(16),
          topRight: const Radius.circular(16),
          bottomLeft: Radius.circular(isUser ? 16 : 4),
          bottomRight: Radius.circular(isUser ? 4 : 16),
        ),
        border: Border.all(
          color: borderColor,
          width: isSelected ? 1.8 : 1.0,
        ),
      ),
      child: Column(
        crossAxisAlignment:
            isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
            if (message.attachment != null && message.attachment!.type != AttachmentType.image)
              Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surface.withValues(alpha: 0.8),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color: message.attachment!.type == AttachmentType.audio
                          ? appColors.audioAccent.withValues(alpha: 0.45)
                          : appColors.documentAccent.withValues(alpha: 0.45),
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        message.attachment!.type == AttachmentType.audio
                            ? Icons.audiotrack_rounded
                            : Icons.description_rounded,
                        color: message.attachment!.type == AttachmentType.audio
                            ? appColors.audioAccent
                            : appColors.documentAccent,
                        size: 24,
                      ),
                      const SizedBox(width: 8),
                      Flexible(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(
                              message.attachment!.name,
                              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
                              overflow: TextOverflow.ellipsis,
                            ),
                            Text(
                              message.attachment!.sizeFormatted,
                              style: TextStyle(fontSize: 11, color: theme.colorScheme.outline),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              )
            else if (message.imageBytes != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(10.0),
                  child: Image.memory(
                    message.imageBytes!,
                    width: 200,
                    fit: BoxFit.cover,
                  ),
                ),
              ),
            if (hasThinking)
              Padding(
                padding: EdgeInsets.only(
                  bottom: message.content.isNotEmpty ? 8.0 : 0.0,
                ),
                child: ThinkingBlockWidget(
                  thinkingContent: message.thinkingContent,
                  isThinking: message.isThinking,
                  isInteractive: !isSelectionMode,
                ),
              ),
            if (isWaitingFirstToken)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 4.0),
                child: SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              )
            else if (isUser || message.content.isNotEmpty)
              GptMarkdown(
                message.content,
                style: TextStyle(
                  color: isUser
                      ? theme.colorScheme.onPrimaryContainer
                      : theme.colorScheme.onSurface,
                  fontSize: 15,
                  height: 1.4,
                ),
                onLinkTap: isSelectionMode
                    ? null
                    : (url, _) => _handleLinkTap(context, url),
              ),
            if (!isUser && message.content.isNotEmpty)
              Align(
                alignment: Alignment.centerRight,
                child: Padding(
                  padding: const EdgeInsets.only(top: 4.0),
                  child: InkWell(
                    borderRadius: BorderRadius.circular(12),
                    onTap: isSelectionMode
                        ? null
                        : () => _copyToClipboard(context, message.content),
                    child: Padding(
                      padding: const EdgeInsets.all(4.0),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.copy_rounded,
                            size: 14,
                            color: theme.colorScheme.outline,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            'Copiar',
                            style: TextStyle(
                              fontSize: 11,
                              color: theme.colorScheme.outline,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
          ],
      ),
    );

    if (!isSelectionMode) {
      return Align(
        alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
        child: GestureDetector(
          onLongPress: isSelectable ? onLongPress : null,
          child: bubbleContainer,
        ),
      );
    }

    return SelectionContainer.disabled(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Material(
          color: isSelected
              ? theme.colorScheme.primaryContainer.withValues(alpha: 0.25)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
          child: InkWell(
            borderRadius: BorderRadius.circular(12),
            onTap: isSelectable ? onToggleSelection : null,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Checkbox(
                    value: isSelected,
                    onChanged:
                        isSelectable ? (_) => onToggleSelection?.call() : null,
                  ),
                  Expanded(
                    child: Align(
                      alignment:
                          isUser ? Alignment.centerRight : Alignment.centerLeft,
                      child: bubbleContainer,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSystemMessage(BuildContext context) {
    final theme = Theme.of(context);
    final isError = message.text.toLowerCase().contains('erro');
    final displayText = message.text.replaceFirst('Sistema:', '').trim();

    return Center(
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isError
              ? theme.colorScheme.errorContainer.withValues(alpha: 0.5)
              : theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.6),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isError
                ? theme.colorScheme.error.withValues(alpha: 0.3)
                : theme.colorScheme.outlineVariant.withValues(alpha: 0.4),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              isError ? Icons.error_outline : Icons.info_outline,
              size: 15,
              color: isError
                  ? theme.colorScheme.error
                  : theme.colorScheme.onSurfaceVariant,
            ),
            const SizedBox(width: 6),
            Flexible(
              child: Text(
                displayText,
                style: TextStyle(
                  fontSize: 12,
                  color: isError
                      ? theme.colorScheme.onErrorContainer
                      : theme.colorScheme.onSurfaceVariant,
                  fontWeight: FontWeight.w500,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _handleLinkTap(BuildContext context, String url) async {
    final uri = Uri.tryParse(url);
    if (uri == null) return;
    try {
      if (await canLaunchUrl(uri)) {
        await launchUrl(uri, mode: LaunchMode.externalApplication);
      } else {
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Não foi possível abrir o link: $url')),
          );
        }
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Erro ao abrir link: $e')),
        );
      }
    }
  }

  void _copyToClipboard(BuildContext context, String text) {
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Mensagem copiada para a área de transferência!'),
        duration: Duration(seconds: 2),
      ),
    );
  }
}
