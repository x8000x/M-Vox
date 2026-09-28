import { pipeline, AutoModelForCTC, AutoProcessor, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

// -----------------------------------------------------------------------------
// Import note
// This line brings in the Hugging Face Transformers library from a CDN so the
// browser can load the speech recognition model at runtime. In simple terms,
// the app is asking the browser to download and use the Whisper engine on the
// fly instead of relying on a server-side backend.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// App overview
// This file keeps the transcription experience organized in a few clear areas:
// UI references, theme and personalization, transcript rendering, audio decoding,
// model loading, and the main transcription workflow. The structure below is
// intended to make the file easier to read without changing how the app works.
// -----------------------------------------------------------------------------

const audioInput = document.getElementById('audioInput');
const transcribeBtn = document.getElementById('transcribeBtn');
const statusEl = document.getElementById('status');
const debugInfo = document.getElementById('debugInfo');
const transcriptEl = document.getElementById('transcript');
const clearBtn = document.getElementById('clearBtn');
const downloadBtn = document.getElementById('downloadBtn');
const audioPlayer = document.getElementById('audioPlayer');
const progressContainer = document.getElementById('progressContainer');
const progressLabel = document.getElementById('progressLabel');
const progressState = document.getElementById('progressState');
const progressBar = document.getElementById('progressBar');
const modelSelect = document.getElementById('modelSelect');
const modelStatus = document.getElementById('modelStatus');
const languageSelect = document.getElementById('languageSelect');
const languageStatus = document.getElementById('languageStatus');
const autoTranslateToggle = document.getElementById('autoTranslateToggle');
const modelDownloadDebug = document.getElementById('modelDownloadDebug');
const modelDownloadName = document.getElementById('modelDownloadName');
const modelDownloadedSize = document.getElementById('modelDownloadedSize');
const modelRemainingSize = document.getElementById('modelRemainingSize');
const modelDownloadBar = document.getElementById('modelDownloadBar');
const modelDownloadState = document.getElementById('modelDownloadState');

// -----------------------------------------------------------------------------
// DOM references and live app state
// These variables point to the HTML controls in the page. They act like a map
// so the script can update buttons, text areas, the player, and the progress
// bar without needing to search the page repeatedly.
// The state variables below keep track of the current file, the loaded model,
// the transcript text, and the audio that has already been prepared.
// -----------------------------------------------------------------------------
let asrPipeline = null;
let currentFile = null;
let audioContext = null;
let currentTranscript = '';
let currentAudioDataPromise = null;
let currentRawAudio = null;
let preloadPipelinePromise = null;
let transcriptionStartTime = null;
let lastTranscriptionElapsed = null;
let lastAudioDuration = null;
let loadedModelId = null;
let pipelineLoadToken = 0;
let highlightProcessor = null;
let highlightModel = null;
let highlightTokenizer = null;
let highlightLoadPromise = null;
const modelDownloadFiles = new Map();

// -----------------------------------------------------------------------------
// Personalization and theme settings
// These values control the look and feel of the page. They store the user's
// preferred color theme, text size, and font so the app feels consistent on
// future visits.
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Preference persistence
// This app saves the user's UI choices in the browser's localStorage. That
// means the selected theme, font size, and font family can be remembered even
// after the page is refreshed or reopened, because the browser keeps that data
// locally on the device.
// -----------------------------------------------------------------------------
const STORAGE_KEY = 'transcriber-personalization';
const MODEL_STORAGE_KEY = 'transcriber-model-selection';
const LANGUAGE_STORAGE_KEY = 'transcriber-language-selection';
const TRANSLATE_STORAGE_KEY = 'transcriber-translate-to-english';
const DEFAULT_MODEL_ID = 'default';
const DEFAULT_LANGUAGE = 'auto';
const LANGUAGE_OPTIONS = [
  ['english', 'English'], ['chinese', 'Chinese'], ['german', 'German'], ['spanish', 'Spanish'],
  ['russian', 'Russian'], ['korean', 'Korean'], ['french', 'French'], ['japanese', 'Japanese'],
  ['portuguese', 'Portuguese'], ['turkish', 'Turkish'], ['polish', 'Polish'], ['catalan', 'Catalan'],
  ['dutch', 'Dutch'], ['arabic', 'Arabic'], ['swedish', 'Swedish'], ['italian', 'Italian'],
  ['indonesian', 'Indonesian'], ['hindi', 'Hindi'], ['finnish', 'Finnish'], ['vietnamese', 'Vietnamese'],
  ['hebrew', 'Hebrew'], ['ukrainian', 'Ukrainian'], ['greek', 'Greek'], ['malay', 'Malay'],
  ['czech', 'Czech'], ['romanian', 'Romanian'], ['danish', 'Danish'], ['hungarian', 'Hungarian'],
  ['tamil', 'Tamil'], ['norwegian', 'Norwegian'], ['thai', 'Thai'], ['urdu', 'Urdu'],
  ['croatian', 'Croatian'], ['bulgarian', 'Bulgarian'], ['lithuanian', 'Lithuanian'], ['latin', 'Latin'],
  ['maori', 'Maori'], ['malayalam', 'Malayalam'], ['welsh', 'Welsh'], ['slovak', 'Slovak'],
  ['telugu', 'Telugu'], ['persian', 'Persian'], ['latvian', 'Latvian'], ['bengali', 'Bengali'],
  ['serbian', 'Serbian'], ['azerbaijani', 'Azerbaijani'], ['slovenian', 'Slovenian'], ['kannada', 'Kannada'],
  ['estonian', 'Estonian'], ['macedonian', 'Macedonian'], ['breton', 'Breton'], ['basque', 'Basque'],
  ['icelandic', 'Icelandic'], ['armenian', 'Armenian'], ['swahili', 'Swahili'], ['galician', 'Galician'],
  ['marathi', 'Marathi'], ['punjabi', 'Punjabi'], ['sinhala', 'Sinhala'], ['khmer', 'Khmer'],
  ['shona', 'Shona'], ['yoruba', 'Yoruba'], ['somali', 'Somali'], ['afrikaans', 'Afrikaans'],
  ['occitan', 'Occitan'], ['georgian', 'Georgian'], ['belarusian', 'Belarusian'], ['tajik', 'Tajik'],
  ['sindhi', 'Sindhi'], ['gujarati', 'Gujarati'], ['amharic', 'Amharic'], ['yiddish', 'Yiddish'],
  ['lao', 'Lao'], ['uzbek', 'Uzbek'], ['faroese', 'Faroese'], ['haitian creole', 'Haitian Creole'],
  ['pashto', 'Pashto'], ['turkmen', 'Turkmen'], ['nynorsk', 'Norwegian Nynorsk'], ['maltese', 'Maltese'],
  ['sanskrit', 'Sanskrit'], ['luxembourgish', 'Luxembourgish'], ['myanmar', 'Myanmar'], ['tibetan', 'Tibetan'],
  ['tagalog', 'Filipino'], ['malagasy', 'Malagasy'], ['assamese', 'Assamese'], ['tatar', 'Tatar'],
  ['hawaiian', 'Hawaiian'], ['lingala', 'Lingala'], ['hausa', 'Hausa'], ['bashkir', 'Bashkir'],
  ['javanese', 'Javanese'], ['sundanese', 'Sundanese'], ['cantonese', 'Cantonese'],
];
LANGUAGE_OPTIONS.sort((left, right) => left[1].localeCompare(right[1]));
const SUPPORTED_LANGUAGES = new Set(['auto', ...LANGUAGE_OPTIONS.map(([value]) => value)]);
const HIGHLIGHT_MODEL_ID = 'onnx-community/mms-300m-1130-forced-aligner-ONNX';
// Loading a second neural model alongside Whisper can exhaust browser memory.
// Whisper timestamps remain available when this optional aligner is disabled.
const USE_CTC_HIGHLIGHT_ALIGNMENT = false;
const MODEL_PROFILES = {
  default: { label: 'Whisper Base', model: 'Xenova/whisper-base', downloadSize: 'about 75 MB' },
  medium: { label: 'Whisper Small', model: 'Xenova/whisper-small', downloadSize: 'about 150 MB' },
  highLite: { label: 'Whisper Large V3 Turbo (q4)', model: 'onnx-community/whisper-large-v3-turbo', dtype: 'q4', downloadSize: 'about 800 MB' },
  high: { label: 'Whisper Large V3 (q4)', model: 'onnx-community/whisper-large-v3-ONNX', dtype: 'q4', downloadSize: 'about 1.5 GB' },
};
const HIGHLIGHT_MODEL_PROFILE = {
  label: 'MMS CTC forced aligner (q8)',
  model: HIGHLIGHT_MODEL_ID,
  dtype: 'q8',
  downloadSize: 'about 340 MB',
};
const DEFAULT_PREFERENCES = { theme: 'blue', fontSize: 'medium', fontFamily: 'Inter' };
const THEME_PRESETS = {
  blue: {
    pageBg: '#07182b',
    surface1: 'rgba(12, 41, 69, 0.9)',
    surface2: 'rgba(7, 26, 47, 0.95)',
    surface3: '#103b5c',
    borderColor: 'rgba(125, 211, 252, 0.24)',
    textPrimary: '#f8fbff',
    textSecondary: '#dbeafe',
    textMuted: '#93c5fd',
    accent: '#38bdf8',
    accentHover: '#0ea5e9',
    accentContrast: '#082f49',
    buttonBg: '#124367',
    buttonText: '#f8fbff',
    highlightBg: 'rgba(125, 211, 252, 0.95)',
    highlightHover: 'rgba(56, 189, 248, 0.18)',
    highlightText: '#052238',
    progressTrack: '#153d61',
    progressFill: '#38bdf8'
  },
  pink: {
    pageBg: '#f8dce4',
    surface1: 'rgba(234, 118, 164, 0.96)',
    surface2: 'rgba(242, 205, 221, 0.98)',
    surface3: '#f0ccd9',
    borderColor: 'rgba(136, 27, 75, 0.28)',
    textPrimary: '#2a0718',
    textSecondary: '#4d1733',
    textMuted: '#6a2743',
    accent: '#c55a8e',
    accentHover: '#f0d1df',
    accentContrast: '#020001',
    buttonBg: '#cc557d',
    buttonText: '#2a0718',
    highlightBg: 'rgba(217, 70, 143, 0.78)',
    highlightHover: 'rgba(217, 70, 143, 0.2)',
    highlightText: '#3f0c23',
    progressTrack: '#e8c1d2',
    progressFill: '#d9468f'
  },
  yellow: {
    pageBg: '#d9c28c',
    surface1: 'rgba(255, 255, 148, 0.96)',
    surface2: 'rgba(244, 225, 188, 0.98)',
    surface3: '#e7cf9b',
    borderColor: 'rgba(120, 53, 15, 0.28)',
    textPrimary: '#241307',
    textSecondary: '#4a2c15',
    textMuted: '#6a3c22',
    accent: '#a25712',
    accentHover: '#fad8bc',
    accentContrast: '#241307',
    buttonBg: '#e9d3a8',
    buttonText: '#241307',
    highlightBg: 'rgba(162, 87, 18, 0.76)',
    highlightHover: 'rgba(162, 87, 18, 0.16)',
    highlightText: '#3c220d',
    progressTrack: '#e9d5ae',
    progressFill: '#a25712'
  },
  green: {
    pageBg: '#123a28',
    surface1: 'rgba(24, 71, 48, 0.9)',
    surface2: 'rgba(13, 45, 32, 0.92)',
    surface3: '#1b5b3b',
    borderColor: 'rgba(74, 222, 128, 0.24)',
    textPrimary: '#f0fdf4',
    textSecondary: '#dcfce7',
    textMuted: '#86efac',
    accent: '#4ade80',
    accentHover: '#22c55e',
    accentContrast: '#052e16',
    buttonBg: '#1f6d45',
    buttonText: '#f0fdf4',
    highlightBg: 'rgba(187, 247, 208, 0.95)',
    highlightHover: 'rgba(74, 222, 128, 0.2)',
    highlightText: '#052e16',
    progressTrack: '#236f42',
    progressFill: '#4ade80'
  },
  black: {
    pageBg: '#030712',
    surface1: 'rgba(17, 24, 39, 0.92)',
    surface2: 'rgba(3, 7, 18, 0.95)',
    surface3: '#111827',
    borderColor: 'rgba(255, 255, 255, 0.16)',
    textPrimary: '#f9fafb',
    textSecondary: '#e5e7eb',
    textMuted: '#9ca3af',
    accent: '#f59e0b',
    accentHover: '#d97706',
    accentContrast: '#111827',
    buttonBg: '#111827',
    buttonText: '#f9fafb',
    highlightBg: 'rgba(245, 158, 11, 0.95)',
    highlightHover: 'rgba(255, 255, 255, 0.1)',
    highlightText: '#111827',
    progressTrack: '#1f2937',
    progressFill: '#f59e0b'
  }
};
const FONT_SIZES = {
  small: { base: '0.92rem', heading: '2rem' },
  medium: { base: '1rem', heading: '2.25rem' },
  large: { base: '1.08rem', heading: '2.5rem' }
};
const FONT_FAMILIES = {
  Inter: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  Poppins: 'Poppins, "Segoe UI", sans-serif',
  Roboto: 'Roboto, "Segoe UI", sans-serif',
  Georgia: 'Georgia, Cambria, "Times New Roman", serif',
  'Segoe UI': '"Segoe UI", Tahoma, sans-serif'
};

let personalization = { ...DEFAULT_PREFERENCES };

const getStoredPreferences = () => {
  // This function reads the saved preferences from the browser.
  // If nothing has been stored yet, it falls back to the default look.
  // The logic is intentionally safe, so a broken or missing saved value does
  // not crash the app.
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    if (stored && typeof stored === 'object') {
      return { ...DEFAULT_PREFERENCES, ...stored };
    }
  } catch (error) {
    console.warn('Unable to read personalization settings:', error);
  }
  return { ...DEFAULT_PREFERENCES };
};

const applyPersonalization = () => {
  // This function applies the saved or selected theme to the page.
  // It updates CSS variables and the visible buttons so the experience feels
  // consistent right away. It also writes the current choices back to storage
  // so future visits keep the same look.
  const theme = THEME_PRESETS[personalization.theme] || THEME_PRESETS.blue;
  const size = FONT_SIZES[personalization.fontSize] || FONT_SIZES.medium;
  const fontFamily = FONT_FAMILIES[personalization.fontFamily] || FONT_FAMILIES.Inter;
  const root = document.documentElement;

  root.style.setProperty('--page-bg', theme.pageBg);
  root.style.setProperty('--surface-1', theme.surface1);
  root.style.setProperty('--surface-2', theme.surface2);
  root.style.setProperty('--surface-3', theme.surface3);
  root.style.setProperty('--border-color', theme.borderColor);
  root.style.setProperty('--text-primary', theme.textPrimary);
  root.style.setProperty('--text-secondary', theme.textSecondary);
  root.style.setProperty('--text-muted', theme.textMuted);
  root.style.setProperty('--accent', theme.accent);
  root.style.setProperty('--accent-hover', theme.accentHover);
  root.style.setProperty('--accent-contrast', theme.accentContrast);
  root.style.setProperty('--button-bg', theme.buttonBg);
  root.style.setProperty('--button-text', theme.buttonText);
  root.style.setProperty('--highlight-bg', theme.highlightBg);
  root.style.setProperty('--highlight-hover', theme.highlightHover);
  root.style.setProperty('--highlight-text', theme.highlightText);
  root.style.setProperty('--progress-track', theme.progressTrack);
  root.style.setProperty('--progress-fill', theme.progressFill);
  root.style.setProperty('--app-font-family', fontFamily);
  root.style.setProperty('--app-font-size', size.base);
  root.style.setProperty('--app-heading-size', size.heading);

  document.body.style.fontFamily = fontFamily;
  document.body.style.fontSize = size.base;

  document.querySelectorAll('.transcript-word.highlight-current').forEach((element) => {
    element.style.color = theme.highlightText;
    element.style.setProperty('color', theme.highlightText, 'important');
  });

  document.querySelectorAll('.theme-option').forEach((button) => {
    const isActive = button.dataset.theme === personalization.theme;
    button.classList.toggle('active', isActive);
    button.style.borderColor = isActive ? 'white' : 'rgba(255,255,255,0.75)';
    button.style.boxShadow = isActive ? '0 0 0 2px rgba(255,255,255,0.9)' : 'none';
  });
  document.querySelectorAll('.font-size-option').forEach((button) => {
    button.classList.toggle('active', button.dataset.size === personalization.fontSize);
  });

  const fontFamilySelect = document.getElementById('fontFamilySelect');
  if (fontFamilySelect) {
    fontFamilySelect.value = personalization.fontFamily;
  }

  try {
    // Saving here is what makes the preferences persistent.
    // The browser stores the value as text in localStorage, which is available
    // to the page later even after a refresh.
    localStorage.setItem(STORAGE_KEY, JSON.stringify(personalization));
  } catch (error) {
    console.warn('Unable to save personalization settings:', error);
  }
};

const toggleCustomizationPanel = () => {
  const panel = document.getElementById('customizePanel');
  const toggle = document.getElementById('customizeToggle');
  if (!panel || !toggle) return;
  const isHidden = panel.classList.toggle('hidden');
  toggle.setAttribute('aria-expanded', String(!isHidden));
};

// -----------------------------------------------------------------------------
// Progress, modal, and status helpers
// These small helpers keep the interface responsive. They update loading bars,
// show helpful messages, and display error popups when something goes wrong.
// -----------------------------------------------------------------------------
const errorModal = document.getElementById('errorModal');
const errorMessage = document.getElementById('errorMessage');
const closeErrorModal = document.getElementById('closeErrorModal');
const retryErrorBtn = document.getElementById('retryErrorBtn');

const showProgress = (label, state, percent = 0) => {
  progressContainer.classList.remove('hidden');
  progressLabel.textContent = label;
  progressState.textContent = state;
  progressBar.style.width = `${Math.min(Math.max(percent, 0), 100)}%`;
};

const hideProgress = () => {
  progressContainer.classList.add('hidden');
  progressBar.style.width = '0%';
  progressLabel.textContent = 'Waiting to start...';
  progressState.textContent = 'Preparing';
};

const formatTime = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) return 'unknown';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = (seconds % 60).toFixed(2);
  const parts = [];
  if (hours) parts.push(`${hours}h`);
  if (minutes) parts.push(`${minutes}m`);
  parts.push(`${secs}s`);
  return parts.join(' ');
};

const updateDebugInfo = () => {
  if (!debugInfo) return;
  const elapsedText = lastTranscriptionElapsed != null ? formatTime(lastTranscriptionElapsed / 1000) : 'pending';
  const audioText = Number.isFinite(lastAudioDuration) ? formatTime(lastAudioDuration) : 'unknown';
  debugInfo.textContent = `Transcribe elapsed: ${elapsedText} · Audio duration: ${audioText}`;
};

let highlightSyncOffsetMs = 0;

const LARGE_FILE_STREAM_THRESHOLD = 90 * 1024 * 1024;
const ASR_CHUNK_SECONDS = 30;
const ASR_CHUNK_SIZE = 16000 * ASR_CHUNK_SECONDS;
const ASR_OVERLAP_SECONDS = 2;
const ASR_OVERLAP_SAMPLES = 16000 * ASR_OVERLAP_SECONDS;
const STREAM_WORKLET_BATCH_SIZE = 65536;
const STREAM_RESAMPLE_SECONDS = 8;
let streamProcessorUrl = null;

const enableDownload = (text) => {
  currentTranscript = text;
  downloadBtn.disabled = !text;
};

const rescaleWordTimingsToPlayback = (wordTimings, decodedDurationSeconds, playbackDurationSeconds) => {
  if (!Number.isFinite(decodedDurationSeconds) || decodedDurationSeconds <= 0
    || !Number.isFinite(playbackDurationSeconds) || playbackDurationSeconds <= 0) {
    console.debug('[transcriber] playback drift correction unavailable', {
      decodedDurationSeconds,
      audioPlayerDuration: playbackDurationSeconds,
    });
    return wordTimings || [];
  }
  const driftRatio = playbackDurationSeconds / decodedDurationSeconds;
  console.debug('[transcriber] playback drift correction', {
    decodedDurationSeconds,
    audioPlayerDuration: playbackDurationSeconds,
    driftRatio,
  });
  if (!Array.isArray(wordTimings) || !wordTimings.length) return [];
  return wordTimings.map((timing) => ({
    ...timing,
    startTimeMs: Math.round(timing.startTimeMs * driftRatio),
    endTimeMs: Math.round(timing.endTimeMs * driftRatio),
    startTime: (timing.startTimeMs * driftRatio) / 1000,
    endTime: (timing.endTimeMs * driftRatio) / 1000,
  }));
};

const waitForAudioMetadata = () => {
  if (Number.isFinite(audioPlayer.duration) && audioPlayer.readyState >= 1) {
    return Promise.resolve(audioPlayer.duration);
  }
  return new Promise((resolve) => {
    audioPlayer.addEventListener('loadedmetadata', () => resolve(audioPlayer.duration), { once: true });
  });
};

const appendFloat32 = (left, right) => {
  if (!left.length) return right;
  if (!right.length) return left;
  const result = new Float32Array(left.length + right.length);
  result.set(left, 0);
  result.set(right, left.length);
  return result;
};

const getSelectedModelId = () => {
  try {
    const storedModelId = localStorage.getItem(MODEL_STORAGE_KEY);
    return MODEL_PROFILES[storedModelId] ? storedModelId : DEFAULT_MODEL_ID;
  } catch (error) {
    return DEFAULT_MODEL_ID;
  }
};

const getModelProfile = (modelId = getSelectedModelId()) => modelId === HIGHLIGHT_MODEL_ID
  ? HIGHLIGHT_MODEL_PROFILE
  : MODEL_PROFILES[modelId] || MODEL_PROFILES[DEFAULT_MODEL_ID];

const formatMegabytes = (bytes) => Number.isFinite(bytes) ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : 'Unknown';

const resetModelDownloadDebug = (modelId) => {
  modelDownloadFiles.clear();
  const profile = getModelProfile(modelId);
  modelDownloadName.textContent = profile.label;
  modelDownloadedSize.textContent = '0 MB';
  modelRemainingSize.textContent = 'Unknown';
  modelDownloadBar.style.width = '0%';
  modelDownloadState.textContent = 'Waiting for download progress...';
};

const updateModelDownloadDebug = (progress, modelId) => {
  if (!progress || !modelDownloadDebug) return;
  const fileName = progress.file || 'model files';
  const fileState = modelDownloadFiles.get(fileName) || { loaded: 0, total: null };
  if (Number.isFinite(progress.loaded)) fileState.loaded = progress.loaded;
  if (Number.isFinite(progress.total)) fileState.total = progress.total;
  if (progress.status === 'done' && fileState.total != null) fileState.loaded = fileState.total;
  modelDownloadFiles.set(fileName, fileState);

  const totals = [...modelDownloadFiles.values()];
  const downloaded = totals.reduce((sum, file) => sum + (file.loaded || 0), 0);
  const total = totals.some((file) => file.total == null) ? null : totals.reduce((sum, file) => sum + file.total, 0);
  const percentage = total ? Math.min(100, (downloaded / total) * 100) : Number.isFinite(progress.progress) ? progress.progress : 0;
  modelDownloadName.textContent = getModelProfile(modelId).label;
  modelDownloadedSize.textContent = formatMegabytes(downloaded);
  modelRemainingSize.textContent = total == null ? 'Unknown' : formatMegabytes(Math.max(0, total - downloaded));
  modelDownloadBar.style.width = `${percentage}%`;
  modelDownloadState.textContent = progress.status === 'done'
    ? `Downloaded ${formatMegabytes(downloaded)}. Checking remaining model files...`
    : `Downloading ${fileName} (${Math.round(percentage)}%)`;
};

const confirmModelLazyLoad = (modelId) => {
  const profile = getModelProfile(modelId);
  const confirmationKey = `transcriber-model-confirmed-${modelId}`;
  try {
    if (sessionStorage.getItem(confirmationKey) === 'true') return true;
  } catch (error) {
    // Continue with the confirmation if browser storage is unavailable.
  }
  const confirmed = window.confirm(
    `${profile.label} has a browser download of ${profile.downloadSize}. It will be cached locally and can use substantial memory. Download and load it now?`,
  );
  if (confirmed) {
    try {
      sessionStorage.setItem(confirmationKey, 'true');
    } catch (error) {
      // The model can still load when browser storage is unavailable.
    }
  }
  return confirmed;
};

const getSelectedLanguage = () => {
  try {
    const storedLanguage = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    return SUPPORTED_LANGUAGES.has(storedLanguage) ? storedLanguage : DEFAULT_LANGUAGE;
  } catch (error) {
    return DEFAULT_LANGUAGE;
  }
};

const getAutoTranslate = () => {
  try {
    return localStorage.getItem(TRANSLATE_STORAGE_KEY) !== 'false';
  } catch (error) {
    return true;
  }
};

const getTranscriptionOptions = () => {
  const language = getSelectedLanguage();
  const translateToEnglish = getAutoTranslate();
  return {
    // Word-level timestamps make Whisper translation dramatically slower in-browser.
    // Translated output uses evenly distributed fallback timings for highlighting.
    return_timestamps: translateToEnglish ? false : 'word',
    task: translateToEnglish ? 'translate' : 'transcribe',
    ...(language !== DEFAULT_LANGUAGE ? { language } : {}),
    generate_kwargs: {
      condition_on_prev_tokens: false,
      suppress_tokens: [-1],
    },
  };
};

const transcribeAudioChunk = async (audio, pipelineInstance) => {
  const transcriptionOptions = getTranscriptionOptions();
  try {
    return await pipelineInstance(audio, transcriptionOptions);
  } catch (error) {
    const message = error?.message || String(error);
    const lacksCrossAttention = message.includes('cross attentions') || message.includes('output_attentions=True');
    if (!lacksCrossAttention) throw error;

    if (modelStatus) {
      modelStatus.textContent = 'This quantized model does not provide word timestamps; retrying transcription without timestamps.';
    }
    return pipelineInstance(audio, {
      ...transcriptionOptions,
      return_timestamps: false,
    });
  }
};

const normalizeToken = (value) => String(value || '').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
const normalizeForAligner = (word) => String(word || '').toLowerCase().replace(/[^a-z']/g, '');

const mergeChunkPayloads = (payloads) => {
  const mergedWords = [];
  const textParts = [];
  payloads.filter(Boolean).forEach((payload) => {
    const currentWords = payload.words || [];
    let duplicateCount = 0;
    const compareLimit = Math.min(20, mergedWords.length, currentWords.length);
    for (let count = compareLimit; count > 0; count -= 1) {
      const previous = mergedWords.slice(-count).map((word) => normalizeToken(word.text));
      const current = currentWords.slice(0, count).map((word) => normalizeToken(word.text));
      if (previous.every((word, index) => word && word === current[index])) {
        duplicateCount = count;
        break;
      }
    }
    mergedWords.push(...currentWords.slice(duplicateCount));

    const currentText = String(payload.text || '').trim();
    if (currentText) {
      const currentTokens = currentText.split(/\s+/);
      const previousTokens = textParts.join(' ').split(/\s+/).filter(Boolean);
      const maxOverlap = Math.min(20, previousTokens.length, currentTokens.length);
      let textOverlap = 0;
      for (let count = maxOverlap; count > 0; count -= 1) {
        const previous = previousTokens.slice(-count).map(normalizeToken);
        const current = currentTokens.slice(0, count).map(normalizeToken);
        if (previous.every((word, index) => word && word === current[index])) {
          textOverlap = count;
          break;
        }
      }
      textParts.push(currentTokens.slice(textOverlap).join(' '));
    }
  });
  return { text: textParts.join(' ').replace(/\s+/g, ' ').trim(), words: mergedWords };
};

const mergeToMono = (buffer) => {
  const channelCount = buffer.numberOfChannels;
  if (channelCount === 1) return buffer.getChannelData(0);
  const length = buffer.length;
  const mono = new Float32Array(length);
  for (let c = 0; c < channelCount; c += 1) {
    const channel = buffer.getChannelData(c);
    for (let i = 0; i < length; i += 1) {
      mono[i] += channel[i];
    }
  }
  for (let i = 0; i < length; i += 1) {
    mono[i] /= channelCount;
  }
  return mono;
};

const decodeAudioDataWithRetry = async (context, audioBuffer) => {
  try {
    return await context.decodeAudioData(audioBuffer);
  } catch (error) {
    console.warn('decodeAudioData failed; retrying with a fresh AudioContext:', error);
    await closeAudioContext();
    const retryContext = await ensureAudioContext();
    return await retryContext.decodeAudioData(audioBuffer.slice(0));
  }
};

// -----------------------------------------------------------------------------
// Transcript rendering and highlighting
// This section turns the transcription text into visible words that can be
// clicked and highlighted while audio playback moves forward. The goal is to
// make the transcript feel interactive and easier to follow.
// -----------------------------------------------------------------------------
let transcriptWords = [];
let currentTranscriptText = '';
let activeWordIndex = -1;
let currentTranscriptWordTimings = [];

const escapeHtml = (value) => {
  const div = document.createElement('div');
  div.textContent = value;
  return div.innerHTML;
};

const toMilliseconds = (value) => {
  if (value == null || value === '') return null;
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return null;
  if (numericValue > 100000) return Math.round(numericValue);
  return Math.round(numericValue * 1000);
};

const normalizeWordTiming = (entry) => {
  if (!entry || typeof entry !== 'object') return null;
  const textValue = typeof entry?.text === 'string'
    ? entry.text.trim()
    : (typeof entry?.word === 'string' ? entry.word.trim() : '');
  if (!textValue) return null;

  const timestampCandidate = entry?.timestamp ?? entry?.time ?? entry?.times ?? entry?.timings ?? null;
  let startTimeMs = null;
  let endTimeMs = null;

  if (Array.isArray(timestampCandidate) && timestampCandidate.length >= 2) {
    startTimeMs = toMilliseconds(timestampCandidate[0]);
    endTimeMs = toMilliseconds(timestampCandidate[1]);
  } else if (timestampCandidate && typeof timestampCandidate === 'object') {
    startTimeMs = toMilliseconds(timestampCandidate.start ?? timestampCandidate.startTime ?? timestampCandidate.from ?? timestampCandidate.begin);
    endTimeMs = toMilliseconds(timestampCandidate.end ?? timestampCandidate.endTime ?? timestampCandidate.to ?? timestampCandidate.finish);
  } else {
    startTimeMs = toMilliseconds(entry?.start ?? entry?.startTime ?? entry?.from ?? entry?.begin);
    endTimeMs = toMilliseconds(entry?.end ?? entry?.endTime ?? entry?.to ?? entry?.finish);
  }

  if (startTimeMs == null && endTimeMs != null) {
    startTimeMs = Math.max(0, endTimeMs - 1000);
  }
  if (endTimeMs == null && startTimeMs != null && entry?.duration != null) {
    endTimeMs = startTimeMs + Math.round(toMilliseconds(entry.duration));
  }

  if (startTimeMs == null || endTimeMs == null || endTimeMs <= startTimeMs) {
    return null;
  }

  return {
    text: textValue,
    startTimeMs,
    endTimeMs,
  };
};

const extractWordTimingsFromResult = (result) => {
  const collected = [];
  const addEntry = (entry) => {
    const normalized = normalizeWordTiming(entry);
    if (normalized) collected.push(normalized);
  };

  if (Array.isArray(result?.words)) {
    result.words.forEach(addEntry);
  }

  if (Array.isArray(result?.chunks)) {
    result.chunks.forEach((chunk) => {
      if (Array.isArray(chunk?.words)) {
        chunk.words.forEach(addEntry);
      } else if (chunk?.timestamp || chunk?.time) {
        addEntry(chunk);
      }
      if (Array.isArray(chunk?.segments)) {
        chunk.segments.forEach(addEntry);
      }
    });
  }

  if (!collected.length && Array.isArray(result?.segments)) {
    result.segments.forEach(addEntry);
  }

  return collected;
};

const normalizeTranscriptionPayload = (result, offsetMs = 0) => {
  const text = Array.isArray(result)
    ? result.map((item) => (item?.text ? String(item.text) : '')).filter(Boolean).join(' ')
    : (result?.text ? String(result.text) : '');

  const words = [];
  const addPayloadWords = (payload, payloadOffsetMs = 0) => {
    const timings = extractWordTimingsFromResult(payload);
    timings.forEach((timing) => {
      const startTimeMs = (timing.startTimeMs ?? 0) + payloadOffsetMs;
      const endTimeMs = (timing.endTimeMs ?? startTimeMs + 1000) + payloadOffsetMs;
      words.push({
        ...timing,
        startTimeMs,
        endTimeMs,
        startTime: startTimeMs / 1000,
        endTime: endTimeMs / 1000,
      });
    });
  };

  if (Array.isArray(result)) {
    result.forEach((item) => addPayloadWords(item, offsetMs));
  } else {
    addPayloadWords(result, offsetMs);
  }

  return { text, words };
};

const setTranscriptWordsTiming = (duration, explicitTimings = []) => {
  if (!duration || !transcriptWords.length || !Number.isFinite(duration)) return;
  const totalWords = transcriptWords.length;
  const durationMs = duration * 1000;
  transcriptWords.forEach((entry, idx) => {
    const explicitTiming = explicitTimings[idx] ?? null;
    if (explicitTiming?.startTimeMs != null && explicitTiming?.endTimeMs != null) {
      entry.startTimeMs = explicitTiming.startTimeMs;
      entry.endTimeMs = explicitTiming.endTimeMs;
      entry.startTime = entry.startTimeMs / 1000;
      entry.endTime = entry.endTimeMs / 1000;
      return;
    }

    entry.startTimeMs = Math.round((durationMs * idx) / totalWords);
    entry.endTimeMs = Math.round((durationMs * (idx + 1)) / totalWords);
    entry.startTime = entry.startTimeMs / 1000;
    entry.endTime = entry.endTimeMs / 1000;
  });
};

const isElementVisible = (el) => {
  const rect = el.getBoundingClientRect();
  return rect.top >= 0 && rect.bottom <= (window.innerHeight || document.documentElement.clientHeight);
};

const findActiveWordIndex = (currentTime) => {
  if (!transcriptWords.length) return -1;
  const currentTimeMs = currentTime * 1000;
  if (activeWordIndex >= 0 && activeWordIndex < transcriptWords.length) {
    const current = transcriptWords[activeWordIndex];
    if (currentTimeMs >= current.startTimeMs && currentTimeMs < current.endTimeMs) {
      return activeWordIndex;
    }
    if (currentTimeMs >= current.endTimeMs) {
      for (let idx = activeWordIndex + 1; idx < transcriptWords.length; idx += 1) {
        const word = transcriptWords[idx];
        if (currentTimeMs >= word.startTimeMs && currentTimeMs < word.endTimeMs) return idx;
      }
    } else {
      for (let idx = activeWordIndex - 1; idx >= 0; idx -= 1) {
        const word = transcriptWords[idx];
        if (currentTimeMs >= word.startTimeMs && currentTimeMs < word.endTimeMs) return idx;
      }
    }
  }
  for (let idx = 0; idx < transcriptWords.length; idx += 1) {
    const word = transcriptWords[idx];
    if (currentTimeMs >= word.startTimeMs && currentTimeMs < word.endTimeMs) return idx;
  }
  return -1;
};

let highlightFrame = null;
const SHOW_WORD_TIMESTAMPS = true;
let lastHighlightDebugIndex = -1;

const formatWordTimestamp = (milliseconds) => {
  if (!Number.isFinite(milliseconds)) return '';
  const totalSeconds = Math.max(0, milliseconds) / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const millis = Math.floor(milliseconds % 1000).toString().padStart(3, '0');
  return `(${minutes}:${seconds.toString().padStart(2, '0')}.${millis})`;
};

const updateTranscriptHighlights = () => {
  if (!audioPlayer || !transcriptWords.length) return;
  const currentTime = Math.max(0, audioPlayer.currentTime - (highlightSyncOffsetMs / 1000));
  const nextIndex = findActiveWordIndex(currentTime);
  if (nextIndex === activeWordIndex) return;
  if (activeWordIndex >= 0 && transcriptWords[activeWordIndex]) {
    transcriptWords[activeWordIndex].element.classList.remove('highlight-current');
  }
  activeWordIndex = nextIndex;
  if (nextIndex !== lastHighlightDebugIndex) {
    const activeWord = nextIndex >= 0 ? transcriptWords[nextIndex] : null;
    console.debug('[transcriber] highlight sync', {
      mediaTimeMs: Math.round(audioPlayer.currentTime * 1000),
      wordIndex: nextIndex,
      word: activeWord?.text || null,
      wordStartTimeMs: activeWord?.startTimeMs ?? null,
      wordEndTimeMs: activeWord?.endTimeMs ?? null,
      deltaToWordStartMs: activeWord?.startTimeMs != null
        ? Math.round(audioPlayer.currentTime * 1000 - activeWord.startTimeMs)
        : null,
    });
    lastHighlightDebugIndex = nextIndex;
  }
  if (activeWordIndex >= 0 && transcriptWords[activeWordIndex]) {
    const element = transcriptWords[activeWordIndex].element;
    element.classList.add('highlight-current');
    if (!isElementVisible(element)) {
      element.scrollIntoView({ behavior: 'auto', block: 'nearest' });
    }
  }
};

const queueTranscriptHighlightUpdate = () => {
  if (highlightFrame) return;
  highlightFrame = requestAnimationFrame(() => {
    highlightFrame = null;
    updateTranscriptHighlights();
  });
};

let highlightSyncLoopActive = false;

const runHighlightSyncLoop = () => {
  if (!highlightSyncLoopActive) return;
  updateTranscriptHighlights();
  requestAnimationFrame(runHighlightSyncLoop);
};

const startHighlightSyncLoop = () => {
  if (highlightSyncLoopActive) return;
  highlightSyncLoopActive = true;
  requestAnimationFrame(runHighlightSyncLoop);
};

const stopHighlightSyncLoop = () => {
  highlightSyncLoopActive = false;
};

const renderTranscript = (text, duration, wordTimings = []) => {
  currentTranscriptText = text;
  currentTranscriptWordTimings = Array.isArray(wordTimings) ? wordTimings : [];
  transcriptWords = [];
  activeWordIndex = -1;
  lastHighlightDebugIndex = -1;
  const tokens = text.match(/(\s+|[^\s]+)/g) || [];
  const fragment = document.createDocumentFragment();
  let wordIndex = 0;

  tokens.forEach((token) => {
    if (/^\s+$/.test(token)) {
      const textNode = document.createTextNode(token.replace(/\n/g, '\n'));
      fragment.appendChild(textNode);
      return;
    }

    const looksLikeWord = /[A-Za-z0-9]/.test(token);
    if (!looksLikeWord) {
      const textNode = document.createTextNode(token);
      fragment.appendChild(textNode);
      return;
    }

    const span = document.createElement('span');
    span.className = 'transcript-word';
    span.dataset.wordIndex = String(wordIndex);
    span.textContent = token;
    fragment.appendChild(span);

    const explicitTiming = currentTranscriptWordTimings[wordIndex] ?? null;
    transcriptWords.push({
      element: span,
      text: token,
      startTime: explicitTiming?.startTimeMs != null ? explicitTiming.startTimeMs / 1000 : null,
      endTime: explicitTiming?.endTimeMs != null ? explicitTiming.endTimeMs / 1000 : null,
      startTimeMs: explicitTiming?.startTimeMs ?? null,
      endTimeMs: explicitTiming?.endTimeMs ?? null,
    });
    const renderedWordIndex = wordIndex;
    span.addEventListener('click', () => {
      const entry = transcriptWords[renderedWordIndex];
      if (entry?.startTimeMs != null) {
        audioPlayer.currentTime = entry.startTimeMs / 1000;
        audioPlayer.play();
      }
    });
    wordIndex += 1;
  });

  if (!transcriptWords.length) {
    const emptySpan = document.createElement('span');
    emptySpan.className = 'text-slate-500';
    emptySpan.textContent = 'No transcription returned.';
    fragment.appendChild(emptySpan);
  }

  transcriptEl.innerHTML = '';
  transcriptEl.appendChild(fragment);
  setTranscriptWordsTiming(duration, currentTranscriptWordTimings);
  transcriptWords.forEach((entry) => {
    const timestampPrefix = SHOW_WORD_TIMESTAMPS && entry.startTimeMs != null
      ? `${formatWordTimestamp(entry.startTimeMs)} `
      : '';
    entry.element.textContent = `${timestampPrefix}${entry.text}`;
  });
  updateTranscriptHighlights();
};

const clearTranscript = () => {
  transcriptWords = [];
  currentTranscriptText = '';
  currentTranscriptWordTimings = [];
  activeWordIndex = -1;
  transcriptEl.innerHTML = '';
};

const isMp3File = (file) => {
  return /\/mpeg$/i.test(file.type) || /\.mp3$/i.test(file.name);
};

// -----------------------------------------------------------------------------
// Audio decoding utilities
// These helpers prepare raw audio for transcription. They convert files into a
// simple format the speech model can process, including handling MP3 files and
// audio resampling when needed.
// -----------------------------------------------------------------------------
const skipId3v2Tag = (view) => {
  if (view[0] === 0x49 && view[1] === 0x44 && view[2] === 0x33) {
    const size = ((view[6] & 0x7f) << 21) |
                 ((view[7] & 0x7f) << 14) |
                 ((view[8] & 0x7f) << 7) |
                 (view[9] & 0x7f);
    return 10 + size;
  }
  return 0;
};

const isMp3Sync = (view, idx) => {
  return idx + 1 < view.length && view[idx] === 0xff && (view[idx + 1] & 0xe0) === 0xe0;
};

const findPrevMp3Sync = (view, end, minStart) => {
  for (let i = end; i > minStart; i--) {
    if (isMp3Sync(view, i)) return i;
  }
  return -1;
};

const findNextMp3Sync = (view, start, maxSearch) => {
  for (let i = start; i < Math.min(view.length - 1, maxSearch); i++) {
    if (isMp3Sync(view, i)) return i;
  }
  return -1;
};

const mergeDecodedAudioBuffers = (buffers) => {
  const converted = [];
  let totalLength = 0;
  for (const buffer of buffers) {
    const monoData = mergeToMono(buffer);
    const resampled = resampleAudio(monoData, buffer.sampleRate, 16000);
    converted.push(resampled);
    totalLength += resampled.length;
  }
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of converted) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
};

const decodeLargeMp3File = async (file) => {
  const arrayBuffer = await file.arrayBuffer();
  const view = new Uint8Array(arrayBuffer);
  let offset = skipId3v2Tag(view);
  const sliceSize = 32 * 1024 * 1024;
  const decodedBuffers = [];
  let context = await ensureAudioContext();

  while (offset < view.length) {
    let end = Math.min(offset + sliceSize, view.length);
    if (end < view.length) {
      const sync = findPrevMp3Sync(view, end, offset + 2);
      if (sync > offset) {
        end = sync;
      } else {
        const nextSync = findNextMp3Sync(view, end, end + 65536);
        if (nextSync > offset) end = nextSync;
      }
    }
    if (end <= offset) end = view.length;
    const slice = arrayBuffer.slice(offset, end);
    offset = end;

    let decoded;
    try {
      decoded = await decodeAudioDataWithRetry(context, slice);
    } catch (error) {
      throw new Error(`MP3 slice decode failed on segment ${decodedBuffers.length + 1} / ${Math.ceil(view.length / sliceSize)}: ${error?.message || error}`);
    }
    decodedBuffers.push(decoded);
    context = audioContext;
  }

  return mergeDecodedAudioBuffers(decodedBuffers);
};

// -----------------------------------------------------------------------------
// Large-file streaming helpers
// These functions are used for very large audio files. Instead of loading the
// whole file at once, the app processes it in smaller pieces so it can stay
// responsive and avoid memory issues.
// -----------------------------------------------------------------------------
const getStreamCaptureProcessorUrl = () => {
  if (streamProcessorUrl) return streamProcessorUrl;
  const processorCode = `class StreamCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.buffer = new Float32Array(0);
      this.threshold = ${STREAM_WORKLET_BATCH_SIZE};
    }

    process(inputs) {
      const input = inputs[0];
      if (input && input[0] && input[0].length) {
        const channelData = input[0];
        const nextBuffer = new Float32Array(this.buffer.length + channelData.length);
        nextBuffer.set(this.buffer);
        nextBuffer.set(channelData, this.buffer.length);
        this.buffer = nextBuffer;

        while (this.buffer.length >= this.threshold) {
          const chunk = new Float32Array(this.buffer.subarray(0, this.threshold));
          this.port.postMessage(chunk, [chunk.buffer]);
          this.buffer = this.buffer.subarray(this.threshold);
        }
      }
      return true;
    }
  }
  registerProcessor('stream-capture-processor', StreamCaptureProcessor);`;
  streamProcessorUrl = URL.createObjectURL(new Blob([processorCode], { type: 'application/javascript' }));
  return streamProcessorUrl;
};

const streamTranscribeLargeFile = async (file, pipelineInstance, updateProgress = () => {}) => {
  // This path is used for very large files. Instead of loading everything into
  // memory at once, the app streams the audio through the browser's audio APIs
  // and processes it in smaller pieces. This helps keep the page responsive for
  // large uploads.
  const audio = document.createElement('audio');
  audio.src = URL.createObjectURL(file);
  audio.preload = 'auto';
  audio.muted = true;
  audio.crossOrigin = 'anonymous';
  audio.style.display = 'none';
  document.body.appendChild(audio);

  const ctx = await ensureAudioContext();
  await ctx.resume();
  await ctx.audioWorklet.addModule(getStreamCaptureProcessorUrl());

  const source = ctx.createMediaElementSource(audio);
  const worklet = new AudioWorkletNode(ctx, 'stream-capture-processor');
  source.connect(worklet).connect(ctx.destination);

  let rawBuffer = new Float32Array(0);
  let resampledBuffer = new Float32Array(0);
  const chunkTexts = [];
  const chunkWordTimings = [];
  let chunkCounter = 0;
  let queue = Promise.resolve();
  let decodeError = null;
  let streamedOffsetMs = 0;
  let decodedSampleCount = 0;

  const enqueueChunk = (chunk) => {
    decodedSampleCount += chunk.length;
    queue = queue.then(async () => {
      const chunkIndex = ++chunkCounter;
      const chunkDurationMs = Math.round((chunk.length / 16000) * 1000);
      updateProgress(`Transcribing chunk ${chunkIndex}`, `Processed ${chunkIndex} chunks`, 55 + Math.min(35, chunkIndex * 2));
      const result = await transcribeAudioChunk(chunk, pipelineInstance);
      const normalized = normalizeTranscriptionPayload(result, streamedOffsetMs);
      if (normalized.text) {
        chunkTexts.push(normalized.text);
      }
      if (normalized.words.length) {
        chunkWordTimings.push(...normalized.words);
      }
      streamedOffsetMs += chunkDurationMs;
    });
  };

  const flushResampledBuffer = () => {
    while (resampledBuffer.length >= ASR_CHUNK_SIZE) {
      const chunk = resampledBuffer.subarray(0, ASR_CHUNK_SIZE);
      resampledBuffer = resampledBuffer.subarray(ASR_CHUNK_SIZE);
      enqueueChunk(chunk);
    }
  };

  worklet.port.onmessage = (event) => {
    rawBuffer = appendFloat32(rawBuffer, event.data);
    if (rawBuffer.length >= ctx.sampleRate * STREAM_RESAMPLE_SECONDS) {
      const resampled = resampleAudio(rawBuffer, ctx.sampleRate, 16000);
      resampledBuffer = appendFloat32(resampledBuffer, resampled);
      rawBuffer = new Float32Array(0);
      flushResampledBuffer();
    }
  };

  audio.onerror = (event) => {
    decodeError = new Error('Audio playback failed during stream decode.');
  };

  const finish = async () => {
    if (decodeError) throw decodeError;
    if (rawBuffer.length) {
      const resampled = resampleAudio(rawBuffer, ctx.sampleRate, 16000);
      resampledBuffer = appendFloat32(resampledBuffer, resampled);
      rawBuffer = new Float32Array(0);
    }
    flushResampledBuffer();
    if (resampledBuffer.length) {
      enqueueChunk(resampledBuffer);
      resampledBuffer = new Float32Array(0);
    }
    await queue;
    return { text: chunkTexts.join(' '), words: chunkWordTimings, decodedDurationSeconds: decodedSampleCount / 16000 };
  };

  try {
    await audio.play();
    await new Promise((resolve, reject) => {
      audio.addEventListener('ended', resolve, { once: true });
      audio.addEventListener('error', () => reject(new Error('Audio playback error during stream decode.')), { once: true });
    });
    const result = await finish();
    return result;
  } finally {
    source.disconnect();
    worklet.disconnect();
    audio.pause();
    audio.src = '';
    audio.remove();
  }
};

const resampleAudio = (input, inputRate, outputRate) => {
  if (inputRate === outputRate) return input;
  const ratio = inputRate / outputRate;
  const outputLength = Math.round(input.length / ratio);
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const t = i * ratio;
    const index = Math.floor(t);
    const nextIndex = Math.min(index + 1, input.length - 1);
    const mix = t - index;
    output[i] = input[index] * (1 - mix) + input[nextIndex] * mix;
  }
  return output;
};

const closeAudioContext = async () => {
  if (!audioContext) return;
  try {
    await audioContext.close();
  } catch (error) {
    console.warn('Failed to close existing AudioContext:', error);
  }
  audioContext = null;
};

const ensureAudioContext = async () => {
  if (audioContext) return audioContext;
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  return audioContext;
};

// -----------------------------------------------------------------------------
// Model loading and audio preparation
// This section loads the Whisper model once and keeps it ready for future
// transcriptions. It also prepares the selected audio so the main workflow can
// start quickly.
// -----------------------------------------------------------------------------
const preloadPipeline = async (device, modelId = getSelectedModelId()) => {
  // This function loads the speech recognition model once and reuses it.
  // The model is not built into the browser; it is fetched from the Hugging Face
  // package and then used locally in the page. That is why the first run can
  // take a little longer than later runs.
  if (asrPipeline && loadedModelId === modelId) return asrPipeline;
  if (asrPipeline && loadedModelId !== modelId) {
    asrPipeline = null;
    loadedModelId = null;
  }
  if (!preloadPipelinePromise) {
    const profile = getModelProfile(modelId);
    const loadToken = ++pipelineLoadToken;
    resetModelDownloadDebug(modelId);
    modelDownloadDebug.open = true;
    const loadPromise = pipeline('automatic-speech-recognition', profile.model, {
      device,
      ...(profile.dtype ? { dtype: profile.dtype } : {}),
      progress_callback: (progress) => updateModelDownloadDebug(progress, modelId),
    });
    let timeoutId;
    const timeoutPromise = new Promise((resolve, reject) => {
      timeoutId = setTimeout(() => reject(new Error(`${profile.label} did not finish initializing within 5 minutes. This browser may not have enough WebGPU memory. Try Whisper Small or Base.`)), 5 * 60 * 1000);
    });
    preloadPipelinePromise = Promise.race([loadPromise, timeoutPromise])
      .finally(() => clearTimeout(timeoutId))
      .then((loadedPipeline) => {
        if (loadToken === pipelineLoadToken && getSelectedModelId() === modelId) {
          asrPipeline = loadedPipeline;
          loadedModelId = modelId;
        }
        modelDownloadBar.style.width = '100%';
        modelDownloadState.textContent = 'Model initialized and ready. Files are cached by Transformers.js.';
        return loadedPipeline;
      })
      .catch((error) => {
        preloadPipelinePromise = null;
        asrPipeline = null;
        loadedModelId = null;
        modelDownloadState.textContent = `Model initialization failed: ${error?.message || error}`;
        throw error;
      });
  }
  return preloadPipelinePromise;
};

const preloadHighlightModel = async (device) => {
  if (highlightProcessor && highlightModel) return { processor: highlightProcessor, model: highlightModel };
  if (highlightLoadPromise) return highlightLoadPromise;

  resetModelDownloadDebug(HIGHLIGHT_MODEL_ID);
  modelDownloadDebug.open = true;
  highlightLoadPromise = Promise.all([
    AutoProcessor.from_pretrained(HIGHLIGHT_MODEL_ID, {
      progress_callback: (progress) => updateModelDownloadDebug(progress, HIGHLIGHT_MODEL_ID),
    }),
    AutoTokenizer.from_pretrained(HIGHLIGHT_MODEL_ID, {
      progress_callback: (progress) => updateModelDownloadDebug(progress, HIGHLIGHT_MODEL_ID),
    }),
    AutoModelForCTC.from_pretrained(HIGHLIGHT_MODEL_ID, {
      device,
      dtype: HIGHLIGHT_MODEL_PROFILE.dtype,
      progress_callback: (progress) => updateModelDownloadDebug(progress, HIGHLIGHT_MODEL_ID),
    }),
  ])
    .then(([processor, tokenizer, model]) => {
      highlightProcessor = processor;
      highlightTokenizer = tokenizer;
      highlightModel = model;
      modelDownloadBar.style.width = '100%';
      modelDownloadState.textContent = 'Highlight model initialized and ready. Files are cached by Transformers.js.';
      return { processor, model };
    })
    .catch((error) => {
      highlightLoadPromise = null;
      highlightProcessor = null;
      highlightTokenizer = null;
      highlightModel = null;
      throw error;
    });
  return highlightLoadPromise;
};

const tensorValues = (tensor) => tensor?.data ? tensor.data : tensor;

const ctcForcedAlign = (logits, targetIds, blankId, durationSeconds) => {
  const dims = logits.dims || [];
  if (dims.length !== 3 || dims[0] !== 1 || !targetIds.length) {
    console.debug('[transcriber] CTC alignment rejected: invalid logits or target sequence', { dims, targetTokenCount: targetIds.length });
    return [];
  }
  const values = tensorValues(logits);
  const channelsFirst = dims[1] === 31;
  const vocabularySize = channelsFirst ? dims[1] : dims[2];
  const frameCount = channelsFirst ? dims[2] : dims[1];
  if (!frameCount || !vocabularySize) {
    console.debug('[transcriber] CTC alignment rejected: empty logits', { dims });
    return [];
  }

  // CTC requires a blank between every target symbol. This also handles
  // repeated letters such as the two l characters in "hello" correctly.
  const labels = new Int32Array(targetIds.length * 2 + 1);
  labels.fill(blankId);
  targetIds.forEach((tokenId, index) => {
    labels[index * 2 + 1] = tokenId;
  });
  const labelCount = labels.length;
  if (labelCount > frameCount) {
    console.debug('[transcriber] CTC alignment rejected: transcript needs more frames than audio provides', { frameCount, labelCount, targetTokenCount: targetIds.length });
    return [];
  }
  const trellis = new Float64Array((frameCount + 1) * labelCount);
  const backpointers = new Int32Array((frameCount + 1) * labelCount);
  trellis.fill(-Infinity);
  backpointers.fill(-1);
  trellis[0] = 0;

  const valueAt = (frame, tokenId) => channelsFirst
    ? values[tokenId * frameCount + frame]
    : values[frame * vocabularySize + tokenId];
  const logProbability = (frame, tokenId) => {
    let maxLogit = -Infinity;
    for (let index = 0; index < vocabularySize; index += 1) maxLogit = Math.max(maxLogit, valueAt(frame, index));
    let denominator = 0;
    for (let index = 0; index < vocabularySize; index += 1) denominator += Math.exp(valueAt(frame, index) - maxLogit);
    return valueAt(frame, tokenId) - maxLogit - Math.log(denominator);
  };

  for (let frame = 1; frame <= frameCount; frame += 1) {
    const previousRow = (frame - 1) * labelCount;
    const currentRow = frame * labelCount;
    for (let label = 0; label < labelCount; label += 1) {
      let best = trellis[previousRow + label];
      let previousLabel = label;
      if (label > 0 && trellis[previousRow + label - 1] > best) {
        best = trellis[previousRow + label - 1];
        previousLabel = label - 1;
      }
      if (label > 1 && labels[label] !== blankId && labels[label] !== labels[label - 2]
        && trellis[previousRow + label - 2] > best) {
        best = trellis[previousRow + label - 2];
        previousLabel = label - 2;
      }
      trellis[currentRow + label] = best + logProbability(frame - 1, labels[label]);
      backpointers[currentRow + label] = previousLabel;
    }
  }

  let frame = frameCount;
  let label = labelCount - 1;
  if (trellis[frame * labelCount + label - 1] > trellis[frame * labelCount + label]) label -= 1;
  if (!Number.isFinite(trellis[frame * labelCount + label])) {
    console.debug('[transcriber] CTC alignment rejected: final traceback state is unreachable');
    return [];
  }
  const labelFrames = targetIds.map(() => ({ start: frameCount, end: 0 }));
  while (frame > 0 && label > 0) {
    if (label % 2 === 1) {
      const tokenIndex = (label - 1) / 2;
      labelFrames[tokenIndex].start = Math.min(labelFrames[tokenIndex].start, frame - 1);
      labelFrames[tokenIndex].end = Math.max(labelFrames[tokenIndex].end, frame);
    }
    label = backpointers[frame * labelCount + label];
    if (label < 0) {
      console.debug('[transcriber] CTC alignment rejected: missing traceback pointer');
      return [];
    }
    frame -= 1;
  }

  if (label > 0 || labelFrames.some((span) => span.end <= span.start)) {
    console.debug('[transcriber] CTC alignment rejected: incomplete token spans', {
      remainingLabels: label,
      missingSpans: labelFrames.filter((span) => span.end <= span.start).length,
    });
    return [];
  }

  const frameDurationMs = (durationSeconds * 1000) / frameCount;
  return labelFrames.map((span) => ({
    startTimeMs: Math.round(span.start * frameDurationMs),
    endTimeMs: Math.round(span.end * frameDurationMs),
  }));
};

const alignTranscriptWords = async (audio, text, device) => {
  const { processor, model } = await preloadHighlightModel(device);
  const tokenizer = highlightTokenizer;
  if (!tokenizer) return [];
  const words = text.match(/\S+/g) || [];
  const targetIds = [];
  const wordRanges = [];
  words.forEach((word, index) => {
    const cleanWord = normalizeForAligner(word);
    const ids = Array.from(tokenizer.encode(cleanWord, { add_special_tokens: false }));
    if (!ids.length) return;
    wordRanges.push({ start: targetIds.length, end: targetIds.length + ids.length, index });
    targetIds.push(...ids);
  });
  if (!targetIds.length || wordRanges.length !== words.length) return [];

  const inputs = await processor(audio);
  const output = await model(inputs);
  const logits = output.logits;
  const blankId = tokenizer.convert_tokens_to_ids('<blank>');
  const tokenTimings = ctcForcedAlign(logits, targetIds, Number.isInteger(blankId) ? blankId : 0, audio.length / 16000);
  const alignedWords = wordRanges.map((range) => {
    const spans = tokenTimings.slice(range.start, range.end).filter((span) => span.endTimeMs > span.startTimeMs);
    if (!spans.length) return null;
    const startTimeMs = spans[0].startTimeMs;
    const endTimeMs = spans[spans.length - 1].endTimeMs;
    return { text: words[range.index], startTimeMs, endTimeMs, startTime: startTimeMs / 1000, endTime: endTimeMs / 1000 };
  });
  return alignedWords.some((word) => !word) ? [] : alignedWords;
};

const prepareAudio = async (file) => {
  // This function prepares the selected audio for transcription.
  // The app turns the uploaded file into a simple audio array that the model can
  // understand. It also caches the prepared audio to avoid re-decoding the same
  // file repeatedly during the same session.
  if (!file) {
    currentAudioDataPromise = null;
    currentRawAudio = null;
    return null;
  }
  if (currentRawAudio && currentFile === file) return currentRawAudio;
  if (currentAudioDataPromise && currentFile === file) return currentAudioDataPromise;
  const decodeFunction = file.size > LARGE_FILE_STREAM_THRESHOLD && isMp3File(file)
    ? decodeLargeMp3File
    : decodeAudioFile;
  currentAudioDataPromise = decodeFunction(file)
    .then((rawAudio) => {
      currentRawAudio = rawAudio;
      return rawAudio;
    })
    .catch((error) => {
      currentAudioDataPromise = null;
      currentRawAudio = null;
      throw error;
    });
  return currentAudioDataPromise;
};

const transcribeLargeMp3File = async (file, pipelineInstance, updateProgress = () => {}) => {
  const arrayBuffer = await file.arrayBuffer();
  const view = new Uint8Array(arrayBuffer);
  let offset = skipId3v2Tag(view);
  const sliceSize = 32 * 1024 * 1024;
  let pendingBuffer = new Float32Array(0);
  const texts = [];
  const wordTimings = [];
  let sliceIndex = 0;
  let totalSlices = Math.ceil((view.length - offset) / sliceSize);
  let decodedSampleCount = 0;

  while (offset < view.length) {
    let end = Math.min(offset + sliceSize, view.length);
    if (end < view.length) {
      const sync = findPrevMp3Sync(view, end, offset + 2);
      if (sync > offset) {
        end = sync;
      } else {
        const nextSync = findNextMp3Sync(view, end, end + 65536);
        if (nextSync > offset) end = nextSync;
      }
    }
    if (end <= offset) end = view.length;

    sliceIndex += 1;
    updateProgress(`Decoding slice ${sliceIndex}/${totalSlices}`, `Preparing audio slice ${sliceIndex}`, 20 + Math.round((sliceIndex / totalSlices) * 20));
    const sliceBuffer = arrayBuffer.slice(offset, end);
    await closeAudioContext();
    const context = await ensureAudioContext();
    let decoded;
    try {
      decoded = await context.decodeAudioData(sliceBuffer);
    } catch (error) {
      throw new Error(`MP3 slice decode failed on segment ${sliceIndex}/${totalSlices}: ${error?.message || error}`);
    }

    const monoData = mergeToMono(decoded);
    const resampled = resampleAudio(monoData, decoded.sampleRate, 16000);
    decodedSampleCount += resampled.length;
    pendingBuffer = appendFloat32(pendingBuffer, resampled);

    while (pendingBuffer.length >= ASR_CHUNK_SIZE) {
      const chunk = pendingBuffer.subarray(0, ASR_CHUNK_SIZE);
      pendingBuffer = pendingBuffer.subarray(ASR_CHUNK_SIZE);
      const chunkResult = await transcribeAudioChunk(chunk, pipelineInstance);
      const normalized = normalizeTranscriptionPayload(chunkResult, Math.round((sliceIndex - 1) * sliceSize / 16000 * 1000));
      if (normalized.text) texts.push(normalized.text);
      if (normalized.words.length) wordTimings.push(...normalized.words);
    }

    offset = end;
  }

  if (pendingBuffer.length > 0) {
    const chunkResult = await transcribeAudioChunk(pendingBuffer, pipelineInstance);
    const normalized = normalizeTranscriptionPayload(chunkResult, Math.round((sliceIndex - 1) * sliceSize / 16000 * 1000));
    if (normalized.text) texts.push(normalized.text);
    if (normalized.words.length) wordTimings.push(...normalized.words);
  }
  return { text: texts.join(' '), words: wordTimings, decodedDurationSeconds: decodedSampleCount / 16000 };
};

const formatTranscript = (text) => {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (!normalized) return '';
  const sentences = normalized.match(/[^.!?]+[.!?]?/g) || [normalized];
  const paragraphs = [];

  const wrapSentence = (sentence) => {
    const words = sentence.trim().split(' ');
    const lines = [];
    let current = [];

    for (const word of words) {
      current.push(word);
      if (current.length >= 12 || /[.!?]$/.test(word)) {
        lines.push(current.join(' '));
        current = [];
      }
    }
    if (current.length) lines.push(current.join(' '));
    return lines.join('\n');
  };

  for (const sentence of sentences) {
    const trimmed = sentence.trim();
    if (!trimmed) continue;
    const withEnding = /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
    const capitalized = withEnding.charAt(0).toUpperCase() + withEnding.slice(1);
    paragraphs.push(wrapSentence(capitalized));
  }

  return paragraphs.join('\n\n');
};

const decodeAudioFile = async (file) => {
  const arrayBuffer = await file.arrayBuffer();
  await closeAudioContext();
  const context = await ensureAudioContext();
  const buffer = await decodeAudioDataWithRetry(context, arrayBuffer);
  const monoData = mergeToMono(buffer);
  return resampleAudio(monoData, buffer.sampleRate, 16000);
};

// -----------------------------------------------------------------------------
// UI wiring and user actions
// These listeners connect the controls on the page to the logic above. They
// make the buttons, file picker, and form controls behave like a real app.
// -----------------------------------------------------------------------------
personalization = getStoredPreferences();
applyPersonalization();

if (modelSelect) {
  modelSelect.value = getSelectedModelId();
  modelStatus.textContent = `Selected model: ${getModelProfile().label}`;
  modelSelect.addEventListener('change', (event) => {
    const nextModelId = event.target.value;
    const nextProfile = getModelProfile(nextModelId);
    if (nextModelId !== getSelectedModelId()) {
      const confirmed = window.confirm(`${nextProfile.label} is a large browser download and may use substantial memory. Switch models? The current model will be unloaded.`);
      if (!confirmed) {
        modelSelect.value = getSelectedModelId();
        return;
      }
      if (asrPipeline && loadedModelId !== nextModelId) {
        asrPipeline = null;
        loadedModelId = null;
      }
      pipelineLoadToken += 1;
      preloadPipelinePromise = null;
    }
    try {
      localStorage.setItem(MODEL_STORAGE_KEY, nextModelId);
    } catch (error) {
      console.warn('Unable to save model selection:', error);
    }
    modelStatus.textContent = `Selected model: ${nextProfile.label}. It will be downloaded and cached in this browser when needed.`;
    statusEl.textContent = `${nextProfile.label} selected. Choose Transcribe to load it.`;
  });
}

if (languageSelect) {
  languageSelect.innerHTML = '<option value="auto">Auto Detect</option>';
  LANGUAGE_OPTIONS.forEach(([value, label]) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    languageSelect.appendChild(option);
  });
  languageSelect.value = getSelectedLanguage();
  languageStatus.textContent = `Language: ${languageSelect.options[languageSelect.selectedIndex].text}. ${LANGUAGE_OPTIONS.length} languages available.`;
  languageSelect.addEventListener('change', (event) => {
    const language = SUPPORTED_LANGUAGES.has(event.target.value) ? event.target.value : DEFAULT_LANGUAGE;
    try {
      localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
    } catch (error) {
      console.warn('Unable to save language selection:', error);
    }
    languageSelect.value = language;
    languageStatus.textContent = `Language: ${languageSelect.options[languageSelect.selectedIndex].text}. ${LANGUAGE_OPTIONS.length} languages available.`;
    const languageLabel = languageSelect.options[languageSelect.selectedIndex].text;
    statusEl.textContent = `${languageLabel} selected. Choose Transcribe to apply it.`;
  });
}

if (autoTranslateToggle) {
  autoTranslateToggle.checked = getAutoTranslate();
  autoTranslateToggle.addEventListener('change', (event) => {
    const shouldTranslate = event.target.checked;
    try {
      localStorage.setItem(TRANSLATE_STORAGE_KEY, String(shouldTranslate));
    } catch (error) {
      console.warn('Unable to save translation preference:', error);
    }
    statusEl.textContent = shouldTranslate
      ? 'Translate to English is on. Choose Transcribe to apply it.'
      : 'Translation is off. The transcript will stay in the spoken language.';
  });
}

const customizeToggle = document.getElementById('customizeToggle');
if (customizeToggle) {
  customizeToggle.addEventListener('click', toggleCustomizationPanel);
}

document.querySelectorAll('.theme-option').forEach((button) => {
  button.addEventListener('click', () => {
    personalization.theme = button.dataset.theme || 'blue';
    applyPersonalization();
  });
});

document.querySelectorAll('.font-size-option').forEach((button) => {
  button.addEventListener('click', () => {
    personalization.fontSize = button.dataset.size || 'medium';
    applyPersonalization();
  });
});

const fontFamilySelect = document.getElementById('fontFamilySelect');
if (fontFamilySelect) {
  fontFamilySelect.addEventListener('change', (event) => {
    personalization.fontFamily = event.target.value || 'Inter';
    applyPersonalization();
  });
}

audioInput.addEventListener('change', () => {
  // When the user selects a file, this block updates the current file state and
  // gets the app ready for transcription. It clears old transcript content and
  // prepares the audio in the background so the next button click can begin
  // quickly.
  currentFile = audioInput.files?.[0] ?? null;
  currentRawAudio = null;
  currentAudioDataPromise = null;
  transcribeBtn.disabled = !currentFile;
  if (currentFile) {
    const isLargeFile = currentFile.size > LARGE_FILE_STREAM_THRESHOLD;
    const isLargeMp3 = isLargeFile && isMp3File(currentFile);
    statusEl.textContent = `Selected file: ${currentFile.name} (${Math.round(currentFile.size / 1024)} KB)`;
    audioPlayer.classList.add('hidden');
    audioPlayer.src = '';
    clearTranscript();
    downloadBtn.disabled = true;
    if (!isLargeFile || isLargeMp3) {
      prepareAudio(currentFile).catch((error) => {
        statusEl.textContent = 'Error preparing audio: ' + (error?.message ?? error?.toString());
        transcribeBtn.disabled = true;
        showErrorModal(error);
      });
    } else {
      statusEl.textContent = `Large non-MP3 file selected; streaming decode will be used during transcription.`;
    }
  } else {
    statusEl.textContent = 'Please select an audio file to transcribe.';
  }
});

clearBtn.addEventListener('click', () => {
  clearTranscript();
  statusEl.textContent = 'Transcript cleared. Choose a new audio file to transcribe.';
  audioInput.value = '';
  currentFile = null;
  transcribeBtn.disabled = true;
  downloadBtn.disabled = true;
  audioPlayer.classList.add('hidden');
  audioPlayer.src = '';
  hideProgress();
});

downloadBtn.addEventListener('click', () => {
  if (!currentTranscript) return;
  const blob = new Blob([currentTranscript], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'transcript.txt';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
});

const getPreferredDevice = () => {
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    return 'webgpu';
  }
  return 'wasm';
};

const isSessionAllocationError = (error) => /bad_alloc|can't create a session|cannot create a session|out of memory/i.test(error?.message || String(error));

audioPlayer.addEventListener('timeupdate', queueTranscriptHighlightUpdate);
audioPlayer.addEventListener('seeked', queueTranscriptHighlightUpdate);
audioPlayer.addEventListener('play', startHighlightSyncLoop);
audioPlayer.addEventListener('pause', stopHighlightSyncLoop);
audioPlayer.addEventListener('ended', stopHighlightSyncLoop);
audioPlayer.addEventListener('loadedmetadata', () => {
  if (currentTranscriptText) {
    setTranscriptWordsTiming(audioPlayer.duration, currentTranscriptWordTimings);
    updateTranscriptHighlights();
  }
  lastAudioDuration = Number.isFinite(audioPlayer.duration) ? audioPlayer.duration : lastAudioDuration;
  updateDebugInfo();
});

const showErrorModal = (error) => {
  const message = error?.message || error?.toString() || 'Unknown error';
  const fileInfo = currentFile ? `File: ${currentFile.name} (${Math.round(currentFile.size / 1024 / 1024)} MB)\n\n` : '';
  const details = error?.stack ? `${fileInfo}${message}\n\n${error.stack}` : `${fileInfo}${message}`;
  errorMessage.textContent = details;
  errorModal.classList.remove('hidden');
};

closeErrorModal.addEventListener('click', () => {
  errorModal.classList.add('hidden');
});

retryErrorBtn.addEventListener('click', async () => {
  errorModal.classList.add('hidden');
  if (currentFile) {
    transcribeBtn.click();
  }
});

// -----------------------------------------------------------------------------
// Main transcription workflow
// This is the heart of the app. It gathers the selected file, loads the model,
// decodes the audio, runs transcription, and then displays the finished text.
// -----------------------------------------------------------------------------
transcribeBtn.addEventListener('click', async () => {
  // This is the main journey of the app: select a file, load the model, decode
  // the audio, run transcription, and display the result. It is browser-based,
  // so the experience depends on the current browser environment and the assets
  // loaded into the page. In practice, offline support is limited because the
  // model and runtime libraries are fetched on demand rather than fully cached
  // by a dedicated service worker in this script.
  if (!currentFile) {
    statusEl.textContent = 'No audio file selected.';
    return;
  }

  transcriptionStartTime = performance.now();
  lastTranscriptionElapsed = null;
  lastAudioDuration = null;
  updateDebugInfo();

  transcribeBtn.disabled = true;
  audioInput.disabled = true;
  const modelId = getSelectedModelId();
  const modelProfile = getModelProfile(modelId);
  const translateToEnglish = getAutoTranslate();
  const transcriptionModeLabel = translateToEnglish ? 'translating to English' : 'preserving the spoken language';
  let device = getPreferredDevice();
  const modelAlreadyLoaded = asrPipeline && loadedModelId === modelId;
  if (!modelAlreadyLoaded && !confirmModelLazyLoad(modelId)) {
    statusEl.textContent = `${modelProfile.label} was not loaded.`;
    return;
  }
  statusEl.textContent = `Loading ${modelProfile.label} on ${device.toUpperCase()}, ${transcriptionModeLabel}...`;
  showProgress('Preparing audio', 'Decoding file…', 10);

  try {
    showProgress('Preparing', `Decoding audio and loading ${modelProfile.label}…`, 25);
    const isLargeFile = currentFile.size > LARGE_FILE_STREAM_THRESHOLD;
    const isLargeMp3 = isLargeFile && isMp3File(currentFile);
    const useStreamDecode = isLargeFile && !isLargeMp3;
    let pipelineInstance;
    try {
      pipelineInstance = asrPipeline && loadedModelId === modelId
        ? asrPipeline
        : await preloadPipeline(device, modelId);
    } catch (modelError) {
      if (device !== 'webgpu' || !isSessionAllocationError(modelError)) throw modelError;
      device = 'wasm';
      statusEl.textContent = `WebGPU could not allocate ${modelProfile.label}; retrying on CPU...`;
      pipelineInstance = await preloadPipeline(device, modelId);
    }

    // Decode only after the model has a session. This prevents the decoded PCM
    // buffer and ONNX runtime buffers from competing for memory during startup.
    const rawAudio = useStreamDecode
      ? null
      : currentRawAudio || await (currentAudioDataPromise || prepareAudio(currentFile));
    asrPipeline = pipelineInstance;
    loadedModelId = modelId;
    modelStatus.textContent = `Active model: ${modelProfile.label}. Cached in this browser.`;
    if (USE_CTC_HIGHLIGHT_ALIGNMENT && (!highlightProcessor || !highlightModel)) {
      const loadHighlightModel = confirmModelLazyLoad(HIGHLIGHT_MODEL_ID);
      if (loadHighlightModel) {
        statusEl.textContent = 'Loading the cached CTC highlight model...';
        try {
          await preloadHighlightModel(device);
        } catch (alignmentModelError) {
          console.warn('CTC highlight model unavailable; continuing with Whisper timings:', alignmentModelError);
          modelStatus.textContent = 'CTC highlight model unavailable; Whisper timings will be used.';
        }
      }
    }
    showProgress('Transcribing', 'Running ASR pipeline…', 55);

    let result;
    if (useStreamDecode) {
      statusEl.textContent = 'Large file detected. Streaming decode and transcribing...';
      showProgress('Transcribing', 'Large file decode path in progress...', 60);
      if (isMp3File(currentFile)) {
        result = await transcribeLargeMp3File(currentFile, asrPipeline, (label, state, percent) => showProgress(label, state, percent));
      } else {
        result = await streamTranscribeLargeFile(currentFile, asrPipeline, (label, state, percent) => showProgress(label, state, percent));
      }
    } else {
      const durationSeconds = rawAudio.length / 16000;
      const chunkSeconds = Math.min(90, Math.max(30, Math.ceil(durationSeconds / 8)));
      const chunkSize = chunkSeconds * 16000;
      const chunkStep = chunkSize - ASR_OVERLAP_SAMPLES;
      const transcribeChunks = async (audioArray) => {
        const totalChunks = Math.ceil(audioArray.length / chunkSize);
        const chunks = [];
        for (let offset = 0; offset < audioArray.length; offset += chunkStep) {
          chunks.push({
            audio: audioArray.subarray(offset, Math.min(offset + chunkSize, audioArray.length)),
            offset,
          });
        }

        const concurrency = 1;
        const results = new Array(chunks.length);
        let cursor = 0;

        const worker = async () => {
          while (true) {
            const idx = cursor++;
            if (idx >= chunks.length) break;
            const chunk = chunks[idx];
            const chunkOffsetMs = Math.round((chunk.offset / 16000) * 1000);
            showProgress('Transcribing', `Chunk ${idx + 1} / ${chunks.length}...`, 55 + Math.round((idx / chunks.length) * 35));
            const chunkResult = await transcribeAudioChunk(chunk.audio, asrPipeline);
            const normalized = normalizeTranscriptionPayload(chunkResult, chunkOffsetMs);
            results[idx] = normalized;
          }
        };

        const workers = [];
        for (let i = 0; i < concurrency; i++) workers.push(worker());
        await Promise.all(workers);

        return mergeChunkPayloads(results);
      };

      result = await transcribeChunks(rawAudio);
    }

    let text = '';
    let recognizedWordTimings = [];
    if (Array.isArray(result)) {
      text = result.map(item => item?.text ?? JSON.stringify(item)).join(' ');
    } else if (result?.text) {
      text = result.text;
    } else {
      text = JSON.stringify(result, null, 2);
    }

    if (result?.words?.length) {
      recognizedWordTimings = result.words;
    } else {
      recognizedWordTimings = extractWordTimingsFromResult(result);
    }

    const formattedText = formatTranscript(text || '');
    currentTranscript = formattedText;
    let highlightTimings = recognizedWordTimings;
    let ctcAlignmentApplied = false;
    if (USE_CTC_HIGHLIGHT_ALIGNMENT && rawAudio && formattedText && highlightProcessor && highlightModel) {
      try {
        statusEl.textContent = 'Aligning transcript words for playback highlighting...';
        const alignedTimings = await alignTranscriptWords(rawAudio, formattedText, device);
        const displayWordCount = (formattedText.match(/\S+/g) || []).filter((word) => /[A-Za-z0-9]/.test(word)).length;
        if (alignedTimings.length !== displayWordCount) {
          throw new Error(`CTC alignment returned ${alignedTimings.length} timings for ${displayWordCount} rendered words.`);
        }
        highlightTimings = alignedTimings;
        ctcAlignmentApplied = true;
      } catch (alignmentError) {
        console.warn('CTC highlight alignment failed; retaining ASR timings:', alignmentError);
      }
    }
    enableDownload(formattedText);
    if (currentFile) {
      audioPlayer.src = URL.createObjectURL(currentFile);
      audioPlayer.classList.remove('hidden');
    }
    const decodedDurationSeconds = Number.isFinite(result?.decodedDurationSeconds)
      ? result.decodedDurationSeconds
      : rawAudio?.length / 16000;
    const playbackDurationSeconds = await waitForAudioMetadata();
    highlightTimings = rescaleWordTimingsToPlayback(
      highlightTimings,
      decodedDurationSeconds,
      playbackDurationSeconds,
    );
    // CTC timings are used when the aligner returns a complete word sequence;
    // Whisper timings remain the fallback when CTC is unavailable or declined.
    renderTranscript(formattedText, audioPlayer.duration, highlightTimings);
    lastTranscriptionElapsed = performance.now() - transcriptionStartTime;
    lastAudioDuration = Number.isFinite(audioPlayer.duration) ? audioPlayer.duration : null;
    updateDebugInfo();
    const timingStatus = ctcAlignmentApplied
      ? 'CTC word alignment applied.'
      : 'Whisper word timings used as fallback.';
    statusEl.textContent = `Transcription complete, ${transcriptionModeLabel}. ${timingStatus}`;
    showProgress('Complete', 'Done', 100);
  } catch (error) {
    console.error(error);
    statusEl.textContent = 'Error: ' + (error?.message ?? error?.toString() ?? 'Unknown error');
    transcriptEl.textContent = '';
    showProgress('Error', 'Failed to transcribe', 100);
    showErrorModal(error);
  } finally {
    transcribeBtn.disabled = !currentFile;
    audioInput.disabled = false;
  }
});


statusEl.textContent = 'Choose an audio file to begin transcription.';