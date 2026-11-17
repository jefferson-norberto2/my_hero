import 'package:flutter/material.dart';

/// Collapsible block that displays the model's reasoning ("thinking") tokens
/// inside an assistant chat bubble.
class ThinkingBlockWidget extends StatefulWidget {
  /// Accumulated reasoning text produced by the model.
  final String thinkingContent;

  /// Whether the model is currently streaming reasoning tokens.
  final bool isThinking;

  /// Whether tapping the header toggles expansion.
  final bool isInteractive;

  const ThinkingBlockWidget({
    super.key,
    required this.thinkingContent,
    required this.isThinking,
    this.isInteractive = true,
  });

  @override
  State<ThinkingBlockWidget> createState() => _ThinkingBlockWidgetState();
}

class _ThinkingBlockWidgetState extends State<ThinkingBlockWidget> {
  // Keeps the thinking block collapsed by default so chat history stays compact.
  bool _isExpanded = false;

  String get _headerLabel {
    if (widget.isThinking) {
      return 'Pensando...';
    }
    return _isExpanded ? 'Ocultar raciocínio' : 'Ver raciocínio';
  }

  void _toggleExpanded() {
    setState(() {
      _isExpanded = !_isExpanded;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final trimmedThinking = widget.thinkingContent.trim();

    return Container(
      decoration: BoxDecoration(
        color: colorScheme.surface.withValues(alpha: 0.7),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: colorScheme.primary.withValues(alpha: 0.22),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          InkWell(
            borderRadius: BorderRadius.circular(12),
            onTap: widget.isInteractive ? _toggleExpanded : null,
            child: Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: 10.0,
                vertical: 8.0,
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (widget.isThinking) ...[
                    SizedBox(
                      width: 12,
                      height: 12,
                      child: CircularProgressIndicator(
                        strokeWidth: 1.8,
                        color: colorScheme.primary,
                      ),
                    ),
                    const SizedBox(width: 8),
                  ] else ...[
                    Icon(
                      Icons.psychology_alt_outlined,
                      size: 16,
                      color: colorScheme.primary,
                    ),
                    const SizedBox(width: 6),
                  ],
                  Flexible(
                    child: Text(
                      _headerLabel,
                      style: theme.textTheme.labelMedium?.copyWith(
                        color: colorScheme.primary,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  const SizedBox(width: 4),
                  Icon(
                    _isExpanded
                        ? Icons.keyboard_arrow_up_rounded
                        : Icons.keyboard_arrow_down_rounded,
                    size: 18,
                    color: colorScheme.primary,
                  ),
                ],
              ),
            ),
          ),
          if (_isExpanded) ...[
            Divider(
              height: 1,
              thickness: 1,
              color: colorScheme.outlineVariant.withValues(alpha: 0.5),
            ),
            Padding(
              padding: const EdgeInsets.all(10.0),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxHeight: 220),
                child: SingleChildScrollView(
                  child: Text(
                    trimmedThinking.isEmpty
                        ? 'Iniciando raciocínio...'
                        : trimmedThinking,
                    softWrap: true,
                    style: theme.textTheme.bodySmall?.copyWith(
                      fontSize: 12.5,
                      height: 1.45,
                      color: colorScheme.onSurfaceVariant,
                      fontStyle: trimmedThinking.isEmpty
                          ? FontStyle.italic
                          : FontStyle.normal,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
