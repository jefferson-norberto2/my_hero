import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:dio/dio.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  FlutterGemma.initialize();
  runApp(const GemmaChatApp());
}

class GemmaChatApp extends StatelessWidget {
  const GemmaChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma Local Chat',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const ChatScreen(),
    );
  }
}

class ChatMessage {
  String text;
  final bool isUser;
  final Uint8List? imageBytes; // Holds the image data as bytes

  ChatMessage({required this.text, required this.isUser, this.imageBytes});
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _messageInputController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  final List<Widget> _backendsWidgets = <Widget>[
    const Text('CPU'),
    const Text('GPU'),
  ];
  final List<bool> _backendsSelected = <bool>[true, false];

  bool _isModelLoaded = false;
  bool _isLoading = false;
  bool _isGenerating = false;
  bool _stopGenerationFlag = false;

  // Download state variables
  double _downloadProgress = 0.0;
  String _loadingStatus = '';

  final List<ChatMessage> _chatHistory = [];
  InferenceModel? _activeModel;
  InferenceChat? _activeChat;
  final ImagePicker _imagePicker = ImagePicker();

  @override
  void dispose() {
    _messageInputController.dispose();
    _scrollController.dispose();
    _activeModel?.close();
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

  Future<void> _downloadAndLoadModel([bool is2B = true]) async {
    setState(() {
      _isLoading = true;
      _downloadProgress = 0.0;
      _loadingStatus = 'Checking local storage...';
    });

    // Replace these URLs with your direct download links
    final downloadUrl = is2B
        ? 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm'
        : 'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm';

    final fileName = is2B ? 'gemma_2b.litertlm' : 'gemma_4b.litertlm';

    try {
      // 1. Get the app's internal documents directory
      final directory = await getApplicationDocumentsDirectory();
      final filePath = '${directory.path}/$fileName';
      final file = File(filePath);

      // 2. Download the file if it doesn't exist
      if (!await file.exists()) {
        setState(() => _loadingStatus = 'Starting download...');

        final dio = Dio();
        await dio.download(
          downloadUrl,
          filePath,
          onReceiveProgress: (received, total) {
            if (total != -1) {
              setState(() {
                _downloadProgress = received / total;
                _loadingStatus =
                    'Downloading: ${(received / 1024 / 1024).toStringAsFixed(1)} MB / ${(total / 1024 / 1024).toStringAsFixed(1)} MB';
              });
            }
          },
        );
      }

      // 3. Load the model from the local file
      setState(() {
        _loadingStatus = 'Loading model $fileName into memory...';
        _downloadProgress = 1.0;
      });

      await FlutterGemma.installModel(
        modelType: ModelType.gemmaIt,
      ).fromFile(filePath).install();

      var preferredBackend = _backendsSelected[0]
          ? PreferredBackend.cpu
          : PreferredBackend.gpu;

      _activeModel = await FlutterGemma.getActiveModel(
        maxTokens: 2048,
        preferredBackend: preferredBackend,
        supportImage: true,
      );

      _activeChat = await _activeModel!.createChat(
        supportImage: true,
        systemInstruction:
            'You are a helpful assistant and always reply in Portuguese.',
      );

      setState(() {
        _isModelLoaded = true;
        _isLoading = false;
        _chatHistory.add(
          ChatMessage(
            text:
                "System: Model $fileName loaded successfully from device storage!",
            isUser: false,
          ),
        );
      });
    } catch (e) {
      setState(() => _isLoading = false);
      WidgetsBinding.instance.addPostFrameCallback((_) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Failed to load model: $e')));
      });
    }
  }

  Future<void> _sendMessage() async {
    final text = _messageInputController.text.trim();
    if (text.isEmpty || _activeModel == null) return;

    setState(() {
      // 1. Add the user's message
      _chatHistory.add(ChatMessage(text: text, isUser: true));

      // 2. Add an empty placeholder message for the AI's response
      _chatHistory.add(ChatMessage(text: "", isUser: false));

      _messageInputController.clear();
      _isGenerating = true;
    });

    _scrollToBottom();

    try {
      await _activeChat!.addQueryChunk(Message.text(text: text, isUser: true));

      // 3. Request the response as a Stream
      final responseStream = _activeChat!.generateChatResponseAsync();

      // 4. Listen to the stream and append each chunk as it arrives
      await for (final chunk in responseStream) {
        if (!mounted || _stopGenerationFlag) break;
        setState(() {
          // Append the chunk to the last message in the history
          if (chunk is TextResponse) {
            _chatHistory.last.text += chunk.token;
          }
        });

        _scrollToBottom();
      }
    } catch (e) {
      setState(() {
        _chatHistory.last.text = "System: Error generating response ($e)";
      });
      _scrollToBottom();
    } finally {
      setState(() => _isGenerating = false);
    }
  }

  Future<void> _sendImage() async {
    try {
      final text = _messageInputController.text.trim();
      _messageInputController.clear();

      final XFile? pickedFile = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 80, 
      );

      if (pickedFile == null) return;

      // 1. Leia o arquivo da imagem como bytes (Uint8List)
      final Uint8List imageBytes = await File(pickedFile.path).readAsBytes();

      setState(() {
        // 2. Display the image in the chat UI
        _chatHistory.add(
          ChatMessage(
            text: text.isEmpty ? "O que tem nesta imagem?" : text,
            isUser: true,
            imageBytes: imageBytes,
          ),
        );
        // Add an empty placeholder message for the AI's response
        _chatHistory.add(ChatMessage(text: "", isUser: false));
        _isGenerating = true;
      });

      _scrollToBottom();

      await _activeChat!.addQueryChunk(
        Message(
          text: text.isEmpty ? "O que tem nesta imagem?" : text,
          imageBytes: imageBytes, // Array de bytes da imagem
          isUser: true,
        ),
      );

      final responseStream = _activeChat!.generateChatResponseAsync();

      await for (final chunk in responseStream) {
        if (!mounted || _stopGenerationFlag) break;
        setState(() {
          if (chunk is TextResponse) {
            _chatHistory.last.text += chunk.token;
          }
        });
      }

      _scrollToBottom();
    } on Exception catch (e) {
      setState(() {
        _chatHistory.last.text = "System: Error generating response ($e)";
      });
      _scrollToBottom();
    } finally {
      setState(() => _isGenerating = false);
    }
  }

  void _stopGeneration() {
    if (_isGenerating && _activeChat != null) {
      _stopGenerationFlag = true;
      Future.delayed(const Duration(milliseconds: 500), () {
        _activeChat!.stopGeneration();
        setState(() => _isGenerating = false);
      }).then((value) {
        _stopGenerationFlag = false;
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
          if (_isModelLoaded)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _isGenerating
                  ? null
                  : () {
                      setState(() {
                        _isModelLoaded = false;
                        _activeModel?.close();
                        _activeModel = null;
                        _activeChat = null;
                        _chatHistory.clear();
                      });
                    },
            ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            if (!_isModelLoaded) ...[
              const Text(
                'Select the Gemma model to download/load:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              ToggleButtons(
                onPressed: _isLoading
                    ? null
                    : (int index) {
                        setState(() {
                          for (int i = 0; i < _backendsSelected.length; i++) {
                            _backendsSelected[i] = i == index;
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
                isSelected: _backendsSelected,
                children: _backendsWidgets,
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: _isLoading
                        ? null
                        : () => _downloadAndLoadModel(true),
                    child: const Text('Load 2B'),
                  ),
                  const SizedBox(width: 16),
                  ElevatedButton(
                    onPressed: _isLoading
                        ? null
                        : () => _downloadAndLoadModel(false),
                    child: const Text('Load 4B'),
                  ),
                ],
              ),

              // --- DOWNLOAD PROGRESS UI ---
              if (_isLoading) ...[
                const SizedBox(height: 24),
                Text(
                  _loadingStatus,
                  style: const TextStyle(fontWeight: FontWeight.w500),
                ),
                const SizedBox(height: 8),
                LinearProgressIndicator(value: _downloadProgress),
                const SizedBox(height: 8),
                Text('${(_downloadProgress * 100).toStringAsFixed(1)}%'),
              ],

              const Divider(height: 40),
            ],

            Expanded(
              child: ListView.builder(
                controller: _scrollController,
                itemCount: _chatHistory.length,
                itemBuilder: (context, index) {
                  final message = _chatHistory[index];
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
                                  width: 200, // Limit image size in chat
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

            if (_isGenerating)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 8.0),
                child: LinearProgressIndicator(),
              ),

            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _messageInputController,
                    enabled: _isModelLoaded && !_isGenerating,
                    decoration: const InputDecoration(
                      hintText: 'Type your message...',
                      border: OutlineInputBorder(),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                IconButton(
                  onPressed: _activeModel == null
                      ? null
                      : (_isModelLoaded && !_isGenerating)
                      ? _sendImage
                      : null,
                  color: Colors.teal,
                  icon: Icon(Icons.add_photo_alternate_rounded),
                ),
                const SizedBox(width: 8),
                IconButton(
                  icon: Icon(
                    !_isGenerating ? Icons.send : Icons.stop_circle_rounded,
                  ),
                  color: !_isGenerating ? Colors.teal : Colors.red,
                  onPressed: _activeModel == null
                      ? null
                      : (_isModelLoaded && !_isGenerating)
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
