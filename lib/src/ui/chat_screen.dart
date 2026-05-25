import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:image_picker/image_picker.dart';
import 'package:my_hero/src/controller/chat_controller.dart';
import 'package:my_hero/src/core/di/injections.dart';
import '../models/chat_message.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _messageInputController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  late final ChatController _chatController;

  final List<Widget> _backendsWidgets = <Widget>[
    const Text('CPU'),
    const Text('GPU'),
  ];
  final ImagePicker _imagePicker = ImagePicker();
  
  @override
  void initState() {
    super.initState();
    _chatController = getIt<ChatController>();
  }

  @override
  void dispose() {
    _messageInputController.dispose();
    _scrollController.dispose();
    _chatController.activeModel?.close();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  

  Future<void> _sendMessage([Uint8List? imageBytes]) async {
    var text = _messageInputController.text.trim();
    if (text.isEmpty || _chatController.activeModel == null) return;

      _chatController.chatHistory.add(
        ChatMessage(text: text, isUser: true, imageBytes: imageBytes),
      );

      _chatController.chatHistory.add(ChatMessage(text: "", isUser: false));

      _messageInputController.clear();
      _chatController.isGenerating.value = true;

    _scrollToBottom();

    try {
      if (imageBytes != null) {
        await _chatController.activeChat!.addQueryChunk(
          Message(text: text, imageBytes: imageBytes, isUser: true),
        );
      } else {
        await _chatController.activeChat!.addQueryChunk(
          Message.text(text: text, isUser: true),
        );
      }
      final responseStream = _chatController.activeChat!.generateChatResponseAsync();

      await for (final chunk in responseStream) {
        if (!mounted || _chatController.stopGenerationFlag.value) break;
        
          if (chunk is TextResponse) {
            _chatController.chatHistory.last.text += chunk.token;
          }

        _scrollToBottom();
      }
    } catch (e) {
      if (!_chatController.stopGenerationFlag.value) {
          _chatController.chatHistory.last.text = "System: Error generating response ($e)";
      }
      _scrollToBottom();
    } finally {
      _chatController.isGenerating.value = false;
    }
  }

  Future<void> _sendImage() async {
    try {
      final XFile? pickedFile = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 80,
      );

      if (pickedFile == null) return;

      final Uint8List imageBytes = await File(pickedFile.path).readAsBytes();

      _sendMessage(imageBytes);
    } catch (e) {
      setState(() {
        _chatController.chatHistory.last.text = "System: Error generating response ($e)";
      });
    }
  }

  void _stopGeneration() {
    if (_chatController.isGenerating.value && _chatController.activeChat != null) {
      _chatController.stopGenerationFlag.value = true;
      Future.delayed(const Duration(milliseconds: 500), () {
        _chatController.activeChat!.stopGeneration();
        _chatController.isGenerating.value = false;
      }).then((value) {
        _chatController.stopGenerationFlag.value = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gemma 4 Local'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          if (_chatController.isModelLoaded.value)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _chatController.isGenerating.value
                  ? null
                  : () {
                        _chatController.isModelLoaded.value = false;
                        _chatController.activeModel?.close();
                        _chatController.activeModel = null;
                        _chatController.activeChat = null;
                        _chatController.chatHistory.clear();
                    },
            ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            if (!_chatController.isModelLoaded.value) ...[
              const Text(
                'Select the Gemma model to download/load:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              ToggleButtons(
                onPressed: _chatController.isLoading.value
                    ? null
                    : (int index) {
                        setState(() {
                          for (int i = 0; i < _chatController.backendsSelected.length; i++) {
                            _chatController.backendsSelected[i] = i == index;
                          }
                        });
                      },
                borderRadius: const BorderRadius.all(Radius.circular(20)),
                selectedBorderColor: Colors.green[700],
                selectedColor: Colors.white,
                fillColor: Colors.green[200],
                color: Colors.green[400],
                constraints: const BoxConstraints(
                  minHeight: 28.0,
                  minWidth: 80.0,
                ),
                isSelected: _chatController.backendsSelected,
                children: _backendsWidgets,
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: _chatController.isLoading.value
                        ? null
                        : () => _chatController.downloadAndLoadModel(true),
                    child: const Text('Load 2B'),
                  ),
                  const SizedBox(width: 16),
                  ElevatedButton(
                    onPressed: _chatController.isLoading.value
                        ? null
                        : () => _chatController.downloadAndLoadModel(false),
                    child: const Text('Load 4B'),
                  ),
                ],
              ),

              // --- DOWNLOAD PROGRESS UI ---
              if (_chatController.isLoading.value) ...[
                const SizedBox(height: 24),
                Text(
                  _chatController.loadingStatus.value,
                  style: const TextStyle(fontWeight: FontWeight.w500),
                ),
                const SizedBox(height: 8),
                LinearProgressIndicator(value: _chatController.downloadProgress.value),
                const SizedBox(height: 8),
                Text('${(_chatController.downloadProgress.value * 100).toStringAsFixed(1)}%'),
              ],

              const Divider(height: 40),
            ],

            Expanded(
              child: ListView.builder(
                controller: _scrollController,
                itemCount: _chatController.chatHistory.length,
                itemBuilder: (context, index) {
                  final message = _chatController.chatHistory[index];
                  return Align(
                    alignment: message.isUser
                        ? Alignment.centerRight
                        : Alignment.centerLeft,
                    child: Container(
                      margin: const EdgeInsets.symmetric(vertical: 4),
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: message.isUser
                            ? Colors.teal.shade50
                            : Colors.grey.shade200,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          if (message.imageBytes != null)
                            Padding(
                              padding: const EdgeInsets.only(bottom: 8.0),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(8.0),
                                child: Image.memory(
                                  message.imageBytes!,
                                  width: 200,
                                  fit: BoxFit.cover,
                                ),
                              ),
                            ),
                          Text(message.text),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),

            if (_chatController.isGenerating.value)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 8.0),
                child: LinearProgressIndicator(),
              ),

            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _messageInputController,
                    enabled: _chatController.isModelLoaded.value && !_chatController.isGenerating.value,
                    decoration: const InputDecoration(
                      hintText: 'Type your message...',
                      border: OutlineInputBorder(),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                IconButton(
                  onPressed: _chatController.activeModel == null
                      ? null
                      : (_chatController.isModelLoaded.value && !_chatController.isGenerating.value)
                      ? _sendImage
                      : null,
                  color: Colors.teal,
                  icon: Icon(Icons.add_photo_alternate_rounded),
                ),
                const SizedBox(width: 8),
                IconButton(
                  icon: Icon(
                    !_chatController.isGenerating.value ? Icons.send : Icons.stop_circle_rounded,
                  ),
                  color: !_chatController.isGenerating.value ? Colors.teal : Colors.red,
                  onPressed: _chatController.activeModel == null
                      ? null
                      : (_chatController.isModelLoaded.value && !_chatController.isGenerating.value)
                      ? _sendMessage
                      : _stopGeneration,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
