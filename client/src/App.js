import { useState, useEffect, useRef } from "react";
import {
  Modal,
  Row,
  Col,
  Card,
  Button,
  Form,
  Spinner,
  Badge,
  Toast,
  ToastContainer,
  Container,
  Navbar,
  Nav,
} from "react-bootstrap";
import { Document, Page, pdfjs } from "react-pdf";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//cdn.jsdelivr.net/npm/pdfjs-dist@3.4.120/build/pdf.worker.min.js`;

export default function App() {
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [files, setFiles] = useState([]);
  const [showPreview, setShowPreview] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [numPages, setNumPages] = useState(null);
  const [pdfError, setPdfError] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [password, setPassword] = useState("");
  const [showPasswordInput, setShowPasswordInput] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const [activeTab, setActiveTab] = useState("upload");
  const messagesEndRef = useRef(null);

  const API_BASE_URL =
    process.env.REACT_APP_API_BASE_URL ||
    "https://harshkalia-24-summarise-files.hf.space/";

  // const API_BASE_URL = "http://localhost:8000";

  const showNotification = (message) => {
    setToastMessage(message);
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      showNotification("Please select at least one file");
      return;
    }

    setIsUploading(true);
    try {
      const form = new FormData();
      // Send session ID or let backend generate if empty
      form.append("session_id", sessionId);
      for (const f of files) form.append("files", f);

      const res = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await res.json();

      // Update session ID with what backend returned
      setSessionId(data.session_id);
      setShowPreview(true);
      setSelectedFile(files[0]);
      setChatMessages([]);

      showNotification(
        data.message ||
          `Uploaded ${data.documents_added} documents successfully`
      );
    } catch (error) {
      console.error("Upload error:", error);
      showNotification(error.message || "Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleQuery = async (message = currentMessage) => {
    if (!message.trim()) return;

    setIsSending(true);

    try {
      const userMessage = {
        id: Date.now(),
        sender: "user",
        text: message,
        timestamp: new Date(),
      };

      setChatMessages((prev) => [...prev, userMessage]);
      setCurrentMessage("");

      // Send to API
      const form = new FormData();
      form.append("session_id", sessionId);
      form.append("question", message);

      const res = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Query failed");
      }

      const data = await res.json();

      // Add AI response to chat
      const aiMessage = {
        id: Date.now() + 1,
        sender: "ai",
        text: data.answer,
        sources: data.sources || [],
        timestamp: new Date(),
      };

      setChatMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error("Query error:", error);

      const errorMessage = {
        id: Date.now() + 1,
        sender: "ai",
        text:
          error.message ||
          "Sorry, I encountered an error processing your request. Please try again.",
        timestamp: new Date(),
      };

      setChatMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsSending(false);
    }
  };

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setPdfLoading(true);
    setPdfError(false);
    setShowPasswordInput(false);
    setPassword("");
  };

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setPdfLoading(false);
  };

  const onDocumentLoadError = (error) => {
    console.error("PDF loading error:", error);

    // Check if error is due to password protection
    if (error.name === "PasswordException") {
      setShowPasswordInput(true);
      setPdfError(false);
    } else {
      setPdfError(true);
      setShowPasswordInput(false);
    }

    setPdfLoading(false);
  };

  const handlePasswordSubmit = () => {
    setPdfLoading(true);
    setShowPasswordInput(false);
    setPdfError(false);
  };

  const handleDownload = () => {
    if (!selectedFile) return;

    const url = URL.createObjectURL(selectedFile);
    const a = document.createElement("a");
    a.href = url;
    a.download = selectedFile.name;
    document.body.appendChild(a);
    a.click();

    // Clean up
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  const clearSession = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/clear?session_id=${sessionId}`);
      if (res.ok) {
        setFiles([]);
        setChatMessages([]);
        showNotification("Session cleared successfully");
      } else {
        const errorData = await res.json();
        showNotification(errorData.detail || "Failed to clear session");
      }
    } catch (error) {
      console.error("Clear session error:", error);
      showNotification("Failed to clear session");
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  return (
    <div className="app-container">
      <Navbar bg="white" expand="lg" className="app-navbar shadow-sm">
        <Container>
          <Navbar.Brand className="d-flex align-items-center">
            <div className="brand-icon me-2">
              <i className="bi bi-chat-left-text-fill"></i>
            </div>
            <span className="fw-bold text-primary">DocuChat</span>
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="ms-auto">
              <Nav.Link
                href="https://github.com/yourusername/yourrepo"
                target="_blank"
                className="d-flex align-items-center"
              >
                <i className="bi bi-github me-1"></i> GitHub
              </Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container className="main-content py-4">
        <ToastContainer position="top-end" className="p-3">
          <Toast
            onClose={() => setShowToast(false)}
            show={showToast}
            delay={3000}
            autohide
            className="custom-toast"
          >
            <Toast.Header className="custom-toast-header">
              <strong className="me-auto">Notification</strong>
              <small>just now</small>
            </Toast.Header>
            <Toast.Body className="custom-toast-body">
              {toastMessage}
            </Toast.Body>
          </Toast>
        </ToastContainer>

        <div className="text-center mb-5">
          <h1 className="display-5 fw-bold text-gradient mb-3">
            DocuChat: Intelligent Document Assistant
          </h1>
          <p className="lead text-muted">
            Upload documents and chat with an AI assistant that understands your
            content
          </p>
        </div>

        <Card className="main-card shadow-lg border-0">
          <Card.Body className="p-4">
            <div className="session-section mb-4">
              <div className="d-flex align-items-center justify-content-between mb-3">
                <h5 className="mb-0 fw-semibold text-primary">
                  <i className="bi bi-key me-2"></i>Session Management
                </h5>
                <div className="d-flex gap-2">
                  <Button
                    variant="outline-primary"
                    size="sm"
                    onClick={() => {
                      setSessionId(crypto.randomUUID());
                      setFiles([]);
                      setChatMessages([]);
                      showNotification("New session created");
                    }}
                    className="d-flex align-items-center"
                  >
                    <i className="bi bi-plus-circle me-1"></i>New Session
                  </Button>
                  <Button
                    variant="outline-danger"
                    size="sm"
                    onClick={clearSession}
                    disabled={files.length === 0}
                    className="d-flex align-items-center"
                  >
                    <i className="bi bi-trash me-1"></i>Clear Session
                  </Button>
                </div>
              </div>

              <div className="session-id-container p-3 bg-light rounded">
                <div className="d-flex align-items-center">
                  <span className="text-muted me-2">Session ID:</span>
                  <code className="session-id-text">{sessionId}</code>
                  <Button
                    variant="link"
                    size="sm"
                    className="ms-2 p-0"
                    onClick={() => {
                      navigator.clipboard.writeText(sessionId);
                      showNotification("Session ID copied to clipboard");
                    }}
                  >
                    <i className="bi bi-clipboard"></i>
                  </Button>
                </div>
                <div className="form-text text-muted mt-1">
                  Share this ID to continue this session on another device
                </div>
              </div>
            </div>

            <div className="nav-tabs-container mb-4">
              <Nav
                variant="tabs"
                defaultActiveKey="upload"
                className="custom-tabs"
              >
                <Nav.Item>
                  <Nav.Link
                    eventKey="upload"
                    className={activeTab === "upload" ? "active" : ""}
                    onClick={() => setActiveTab("upload")}
                  >
                    <i className="bi bi-cloud-arrow-up me-1"></i>Upload Files
                  </Nav.Link>
                </Nav.Item>
                {files.length > 0 && (
                  <Nav.Item>
                    <Nav.Link
                      eventKey="preview"
                      className={activeTab === "preview" ? "active" : ""}
                      onClick={() => {
                        setActiveTab("preview");
                        setShowPreview(true);
                      }}
                    >
                      <i className="bi bi-eye me-1"></i>Preview & Chat
                    </Nav.Link>
                  </Nav.Item>
                )}
              </Nav>
            </div>

            {activeTab === "upload" && (
              <div className="upload-section">
                <div className="mb-4">
                  <label className="form-label fw-semibold text-primary">
                    <i className="bi bi-folder-plus me-1"></i>Select Files
                  </label>
                  <div className="d-flex align-items-center">
                    <div className="file-input-container flex-grow-1">
                      <input
                        type="file"
                        className="form-control"
                        multiple
                        onChange={(e) => setFiles([...e.target.files])}
                        accept=".pdf,.jpg,.jpeg,.png,.docx,.txt"
                        id="fileInput"
                      />
                      <label htmlFor="fileInput" className="file-input-label">
                        <i className="bi bi-cloud-arrow-up me-2"></i>
                        Choose files or drag them here
                      </label>
                    </div>
                    <button
                      onClick={handleUpload}
                      className="btn btn-primary ms-3 upload-btn"
                      disabled={isUploading || files.length === 0}
                    >
                      {isUploading ? (
                        <>
                          <Spinner
                            as="span"
                            animation="border"
                            size="sm"
                            role="status"
                            aria-hidden="true"
                            className="me-2"
                          />
                          Processing...
                        </>
                      ) : (
                        <>
                          <i className="bi bi-rocket-takeoff me-2"></i>Process
                          Files
                        </>
                      )}
                    </button>
                  </div>
                  <div className="form-text text-muted mt-2">
                    Supports PDFs, images (JPG, PNG), and text documents
                  </div>
                </div>

                {files.length > 0 && (
                  <div className="selected-files-section">
                    <h6 className="text-primary fw-semibold mb-3">
                      <i className="bi bi-files me-1"></i>Selected Files (
                      {files.length})
                    </h6>
                    <div className="files-grid">
                      {files.map((file, index) => (
                        <div
                          key={index}
                          className="file-item"
                          onClick={() => handleFileSelect(file)}
                        >
                          <div className="file-icon">
                            <i
                              className={`bi ${
                                file.type.startsWith("image/")
                                  ? "bi-file-image"
                                  : "bi-file-earmark-pdf"
                              }`}
                            ></i>
                          </div>
                          <div className="file-details">
                            <div className="file-name">{file.name}</div>
                            <div className="file-size">
                              {Math.round(file.size / 1024)} KB
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === "preview" && (
              <div className="text-center py-4">
                <i
                  className="bi bi-eye-fill text-primary mb-3"
                  style={{ fontSize: "2.5rem" }}
                ></i>
                <h5>Preview & Chat</h5>
                <p className="text-muted">
                  Click the button below to open the document preview and chat
                  interface
                </p>
                <Button
                  variant="primary"
                  onClick={() => setShowPreview(true)}
                  className="mt-2"
                >
                  Open Preview
                </Button>
              </div>
            )}
          </Card.Body>
        </Card>

        {/* Preview Modal */}
        <Modal
          show={showPreview}
          onHide={() => setShowPreview(false)}
          size="xl"
          centered
          backdrop="static"
          className="preview-modal"
          fullscreen="lg-down"
        >
          <Modal.Header closeButton className="modal-header-custom">
            <Modal.Title className="d-flex align-items-center">
              <i className="bi bi-file-earmark-text me-2"></i>Document Preview &
              Chat
            </Modal.Title>
            <div className="session-info">
              <Badge bg="light" text="dark" className="me-2">
                <i className="bi bi-hash me-1"></i>
                {sessionId.slice(0, 8)}...
              </Badge>
              <Badge bg="primary">
                <i className="bi bi-files me-1"></i>
                {files.length}
              </Badge>
            </div>
          </Modal.Header>
          <Modal.Body className="modal-body-custom">
            <Row className="h-100 g-3">
              {/* Left Side - Document Preview */}
              <Col md={6} className="h-100 d-flex flex-column">
                <div className="preview-header mb-3">
                  <h5 className="text-primary">
                    <i className="bi bi-file-earmark me-2"></i>Document Preview
                  </h5>
                  <div className="file-tabs">
                    {files.map((file, index) => (
                      <Button
                        key={index}
                        variant={
                          selectedFile === file ? "primary" : "outline-primary"
                        }
                        size="sm"
                        onClick={() => handleFileSelect(file)}
                        className="me-1 mb-1"
                      >
                        <i
                          className={`bi ${
                            file.type.startsWith("image/")
                              ? "bi-image"
                              : "bi-file-earmark-pdf"
                          } me-1`}
                        ></i>
                        {file.name.length > 15
                          ? `${file.name.substring(0, 15)}...`
                          : file.name}
                      </Button>
                    ))}
                  </div>
                </div>

                <div className="preview-content flex-grow-1 overflow-auto border rounded p-3 bg-light">
                  {selectedFile ? (
                    selectedFile.type.startsWith("image/") ? (
                      <div className="text-center">
                        <img
                          src={URL.createObjectURL(selectedFile)}
                          alt={selectedFile.name}
                          className="img-fluid rounded shadow-sm preview-image"
                        />
                        <div className="mt-3 fw-semibold text-primary">
                          {selectedFile.name}
                        </div>
                      </div>
                    ) : (
                      <div className="w-100 text-center">
                        {pdfError ? (
                          <div className="error-state text-center p-4">
                            <div className="text-danger mb-3">
                              <i
                                className="bi bi-exclamation-triangle-fill"
                                style={{ fontSize: "3rem" }}
                              ></i>
                              <p className="fw-bold mt-2">Failed to load PDF</p>
                            </div>
                            <p className="small mb-3">
                              The file might be encrypted or corrupted
                            </p>
                            <div className="d-flex justify-content-center gap-2 flex-wrap">
                              <Button
                                variant="outline-primary"
                                size="sm"
                                onClick={handleDownload}
                                className="d-flex align-items-center"
                              >
                                <i className="bi bi-download me-1"></i> Download
                              </Button>
                              <Button
                                variant="outline-secondary"
                                size="sm"
                                onClick={() => handleFileSelect(selectedFile)}
                                className="d-flex align-items-center"
                              >
                                <i className="bi bi-arrow-repeat me-1"></i>{" "}
                                Retry
                              </Button>
                            </div>
                            {showPasswordInput && (
                              <div className="mt-3">
                                <p className="small mb-2">
                                  This PDF requires a password
                                </p>
                                <div className="d-flex justify-content-center">
                                  <Form.Control
                                    type="password"
                                    placeholder="Enter password"
                                    value={password}
                                    onChange={(e) =>
                                      setPassword(e.target.value)
                                    }
                                    size="sm"
                                    style={{ maxWidth: "200px" }}
                                  />
                                  <Button
                                    variant="primary"
                                    size="sm"
                                    className="ms-2"
                                    onClick={handlePasswordSubmit}
                                  >
                                    Submit
                                  </Button>
                                </div>
                              </div>
                            )}
                          </div>
                        ) : pdfLoading ? (
                          <div className="d-flex flex-column align-items-center justify-content-center h-100">
                            <Spinner animation="border" variant="primary" />
                            <p className="mt-2">Loading document...</p>
                          </div>
                        ) : (
                          <div className="pdf-container">
                            <Document
                              file={selectedFile}
                              onLoadSuccess={onDocumentLoadSuccess}
                              onLoadError={onDocumentLoadError}
                              password={password}
                              loading={
                                <div className="d-flex flex-column align-items-center justify-content-center h-100">
                                  <Spinner
                                    animation="border"
                                    variant="primary"
                                  />
                                  <p className="mt-2">Loading PDF...</p>
                                </div>
                              }
                            >
                              <Page
                                pageNumber={1}
                                width={400}
                                renderAnnotationLayer={false}
                                renderTextLayer={false}
                              />
                            </Document>
                            {numPages && (
                              <div className="text-center mt-2 text-muted small">
                                Page 1 of {numPages}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  ) : (
                    <div className="text-center text-muted h-100 d-flex flex-column justify-content-center align-items-center">
                      <i
                        className="bi bi-file-earmark"
                        style={{ fontSize: "3rem" }}
                      ></i>
                      <p className="mt-3">Select a file to preview</p>
                    </div>
                  )}
                </div>
              </Col>

              {/* Right Side - Chat Interface */}
              <Col md={6} className="h-100 d-flex flex-column">
                <div className="chat-header mb-3">
                  <h5 className="text-primary">
                    <i className="bi bi-chat-left-text me-2"></i>Chat with your
                    Documents
                  </h5>
                  <div className="chat-info text-muted small">
                    Ask questions about your uploaded content
                  </div>
                </div>

                <div className="chat-messages flex-grow-1 overflow-auto mb-3 border rounded p-3">
                  {chatMessages.length === 0 ? (
                    <div className="text-center text-muted h-100 d-flex flex-column justify-content-center">
                      <div className="mb-3">
                        <i className="bi bi-chat-left-text fs-1 text-primary"></i>
                      </div>
                      <p>Ask questions about your uploaded documents</p>
                      <div className="suggestion-buttons mt-3 d-flex flex-wrap justify-content-center gap-2">
                        <Button
                          variant="outline-primary"
                          size="sm"
                          onClick={() => handleQuery("Summarize this document")}
                        >
                          Summarize
                        </Button>
                        <Button
                          variant="outline-primary"
                          size="sm"
                          onClick={() =>
                            handleQuery("What are the key points?")
                          }
                        >
                          Key Points
                        </Button>
                        <Button
                          variant="outline-primary"
                          size="sm"
                          onClick={() => handleQuery("Explain this content")}
                        >
                          Explain
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="messages-container">
                      {chatMessages.map((message) => (
                        <div
                          key={message.id}
                          className={`message-row ${
                            message.sender === "user"
                              ? "user-message"
                              : "ai-message"
                          }`}
                        >
                          <div className="message-content">
                            <div className="message-sender">
                              {message.sender === "user" ? "You" : "Assistant"}
                              <span className="message-time">
                                {message.timestamp.toLocaleTimeString([], {
                                  hour: "2-digit",
                                  minute: "2-digit",
                                })}
                              </span>
                            </div>
                            <div className="message-text">{message.text}</div>
                            {message.sender === "ai" &&
                              message.sources &&
                              message.sources.length > 0 && (
                                <div className="sources-section mt-2">
                                  <div className="sources-label">Sources:</div>
                                  {message.sources.map((source, idx) => (
                                    <div key={idx} className="source-item">
                                      <div className="source-name">
                                        <i className="bi bi-file-earmark me-1"></i>
                                        {source.filename}
                                        {source.page && (
                                          <span className="ms-1">
                                            (Page {source.page})
                                          </span>
                                        )}
                                      </div>
                                      {source.snippet && (
                                        <div className="source-snippet text-truncate">
                                          {source.snippet}...
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              )}
                          </div>
                        </div>
                      ))}
                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </div>

                <div className="chat-input-container">
                  <Form.Group className="d-flex align-items-center">
                    <Form.Control
                      as="textarea"
                      rows={2}
                      value={currentMessage}
                      onChange={(e) => setCurrentMessage(e.target.value)}
                      placeholder="Ask a question about your documents..."
                      onKeyPress={handleKeyPress}
                      disabled={isSending}
                      className="chat-input rounded-start"
                    />
                    <Button
                      variant="primary"
                      className="chat-send-btn rounded-end"
                      onClick={() => handleQuery()}
                      disabled={isSending || !currentMessage.trim()}
                    >
                      {isSending ? (
                        <Spinner as="span" animation="border" size="sm" />
                      ) : (
                        <i className="bi bi-send-fill"></i>
                      )}
                    </Button>
                  </Form.Group>
                </div>
              </Col>
            </Row>
          </Modal.Body>
          <Modal.Footer className="modal-footer-custom">
            <Button
              variant="outline-secondary"
              onClick={() => setShowPreview(false)}
              className="d-flex align-items-center"
            >
              <i className="bi bi-x-lg me-1"></i>Close Preview
            </Button>
          </Modal.Footer>
        </Modal>
      </Container>

      <footer className="app-footer py-3 mt-5 bg-light border-top">
        <Container>
          <div className="text-center text-muted">
            <small>
              DocuChat &copy; {new Date().getFullYear()} - Intelligent Document
              Analysis
            </small>
          </div>
        </Container>
      </footer>
    </div>
  );
}
