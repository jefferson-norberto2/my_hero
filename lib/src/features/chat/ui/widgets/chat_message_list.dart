import 'package:flutter/material.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_bubble.dart';
import 'package:signals_flutter/signals_flutter.dart';

/// Displays the scrollable list of chat messages and handles selection mode.
class ChatMessageList extends StatefulWidget {
  final ChatController controller;

  const ChatMessageList({super.key, required this.controller});

  @override
  State<ChatMessageList> createState() => _ChatMessageListState();
}

class _ChatMessageListState extends State<ChatMessageList> {
  final ScrollController _scrollController = ScrollController();
  int _lastMessageCount = 0;
  String _lastMessageSignature = '';

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        final messages = widget.controller.chatHistory.value;
        final isSelectionMode = widget.controller.isSelectionMode.value;
        final selectedIndices = widget.controller.selectedMessageIndices.value;

        final lastSignature = messages.isEmpty
            ? ''
            : '${messages.last.content.length}:${messages.last.thinkingContent.length}';

        // Scroll to the bottom only when new messages or streamed tokens arrive,
        // preserving scroll offset while toggling message selections.
        if (messages.length != _lastMessageCount ||
            lastSignature != _lastMessageSignature) {
          _lastMessageCount = messages.length;
          _lastMessageSignature = lastSignature;
          _scrollToBottom();
        }

        return SelectionArea(
          child: ListView.builder(
            controller: _scrollController,
            itemCount: messages.length,
            itemBuilder: (context, index) {
              final isSelectable = widget.controller.isMessageSelectable(index);
              return ChatBubble(
                message: messages[index],
                isSelectionMode: isSelectionMode,
                isSelected: selectedIndices.contains(index),
                isSelectable: isSelectable,
                onLongPress: () => widget.controller.enterSelectionMode(
                  initialMessageIndex: index,
                ),
                onToggleSelection: () =>
                    widget.controller.toggleMessageSelection(index),
              );
            },
          ),
        );
      },
    );
  }
}
