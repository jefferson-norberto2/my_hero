import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gpt_markdown/gpt_markdown.dart';
import 'package:url_launcher/url_launcher.dart';
import '../../data/adapter/chat_message_adapter.dart';

class ChatBubble extends StatelessWidget {
  final ChatMessage message;

  const ChatBubble({super.key, required this.message});

  bool get _isSystemMessage =>
      !message.isUser && message.text.startsWith('Sistema:');

  @override
  Widget build(BuildContext context) {
    if (_isSystemMessage) {
      return _buildSystemMessage(context);
    }

    final isUser = message.isUser;
    final theme = Theme.of(context);
    final maxWidth = MediaQuery.sizeOf(context).width * 0.85;
    final isWaitingFirstToken = !isUser && message.text.isEmpty;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(maxWidth: maxWidth),
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: isUser
              ? theme.colorScheme.primaryContainer.withValues(alpha: 0.6)
              : theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(16),
            topRight: const Radius.circular(16),
            bottomLeft: Radius.circular(isUser ? 16 : 4),
            bottomRight: Radius.circular(isUser ? 4 : 16),
          ),
          border: Border.all(
            color: isUser
                ? theme.colorScheme.primary.withValues(alpha: 0.15)
                : theme.colorScheme.outlineVariant.withValues(alpha: 0.4),
            width: 1,
          ),
        ),
        child: Column(
          crossAxisAlignment:
              isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            if (message.imageBytes != null)
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
            if (isWaitingFirstToken)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 4.0),
                child: SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              )
            else
              GptMarkdown(
                message.text,
                style: TextStyle(
                  color: isUser
                      ? theme.colorScheme.onPrimaryContainer
                      : theme.colorScheme.onSurface,
                  fontSize: 15,
                  height: 1.4,
                ),
                onLinkTap: (url, _) => _handleLinkTap(context, url),
              ),
            if (!isUser && message.text.isNotEmpty)
              Align(
                alignment: Alignment.centerRight,
                child: Padding(
                  padding: const EdgeInsets.only(top: 4.0),
                  child: InkWell(
                    borderRadius: BorderRadius.circular(12),
                    onTap: () => _copyToClipboard(context, message.text),
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
