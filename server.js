// server.js - OpenAI to xKiro API Proxy (adapté depuis openai-nim-proxy)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware - l'ordre compte
app.use(cors()); // <-- c'est CETTE ligne qui ajoute Access-Control-Allow-Origin: *
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// ---- Config xKiro ----
const XKIRO_API_BASE = process.env.XKIRO_API_BASE || 'https://api.xkiro.com/v1';
const XKIRO_API_KEY = process.env.XKIRO_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// ---- Limites internes ----
const RATE_LIMIT_RPM = parseInt(process.env.RATE_LIMIT_RPM || '60', 10);
const WINDOW_MS = 60_000;
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '3', 10);
const REQUEST_TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT_MS || '120000', 10);
const DEFAULT_MAX_TOKENS = parseInt(process.env.DEFAULT_MAX_TOKENS || '4096', 10);

// Mapping des modèles (alias pratiques -> ID xKiro réel avec préfixe vendeur)
// Complète cette liste avec GET https://api.xkiro.com/v1/models
const MODEL_MAPPING = {
  'DEEP4F': 'deepseek/deepseek-v4-flash',
  'DEEP4P': 'deepseek/deepseek-v4-pro',
  'CLAUDE': 'anthropic/claude-opus-5',
  'GPT': 'openai/gpt-5.6-terra',
};

// =====================================================================
//  RATE LIMITER : identique à ta version NIM
// =====================================================================
let requestTimestamps = [];
const queue = [];
let processing = false;

function slotsUsed() {
  const now = Date.now();
  requestTimestamps = requestTimestamps.filter((t) => now - t < WINDOW_MS);
  return requestTimestamps.length;
}

function scheduleXkiroCall(fn) {
  return new Promise((resolve, reject) => {
    queue.push({ fn, resolve, reject });
    processQueue();
  });
}

async function processQueue() {
  if (processing) return;
  processing = true;
  while (queue.length > 0) {
    if (slotsUsed() >= RATE_LIMIT_RPM) {
      const wait = Math.max(WINDOW_MS - (Date.now() - requestTimestamps[0]) + 50, 100);
      console.log(`[rate-limit] file pleine (${queue.length} en attente), pause ${wait}ms`);
      await new Promise((r) => setTimeout(r, wait));
      continue;
    }
    const { fn, resolve, reject } = queue.shift();
    requestTimestamps.push(Date.now());
    fn().then(resolve).catch(reject);
  }
  processing = false;
}

// Appel xKiro avec retry/backoff interne (respecte Retry-After)
async function callXkiro(xkiroRequest, isStream) {
  let attempt = 0;
  while (true) {
    const response = await scheduleXkiroCall(() =>
      axios.post(`${XKIRO_API_BASE}/chat/completions`, xkiroRequest, {
        headers: {
          Authorization: `Bearer ${XKIRO_API_KEY}`,
          'Content-Type': 'application/json',
        },
        responseType: isStream ? 'stream' : 'json',
        timeout: isStream ? 0 : REQUEST_TIMEOUT_MS,
        validateStatus: () => true,
      })
    );

    if (response.status === 429 && attempt < MAX_RETRIES) {
      const retryAfter = parseInt(response.headers['retry-after'] || '0', 10);
      const wait = retryAfter > 0 ? retryAfter * 1000 : Math.min(2 ** attempt * 1000, 8000);
      console.warn(`[429] tentative ${attempt + 1}/${MAX_RETRIES}, nouvelle tentative dans ${wait}ms`);
      if (isStream && response.data && typeof response.data.resume === 'function') {
        response.data.resume();
      }
      await new Promise((r) => setTimeout(r, wait));
      attempt++;
      continue;
    }
    return response;
  }
}

// Résolution du modèle : alias custom -> sinon on laisse passer tel quel
// (utile car les IDs xKiro contiennent déjà le préfixe vendeur, ex: "anthropic/claude-opus-5")
function resolveModel(model) {
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];
  return model; // déjà un identifiant xKiro complet (vendeur/modèle) ou on laisse xKiro trancher
}

// ---- Endpoints ----
app.get('/', (req, res) => {
  res.json({
    service: 'OpenAI to xKiro Proxy',
    version: '1.0.0',
    endpoints: { health: '/health', models: '/v1/models', chat: '/v1/chat/completions' },
  });
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to xKiro Proxy',
    xkiro_api_configured: !!XKIRO_API_KEY,
    rate_limit_rpm: RATE_LIMIT_RPM,
    slots_used: slotsUsed(),
    queue_length: queue.length,
  });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map((model) => ({
      id: model,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'xkiro-proxy',
    })),
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!XKIRO_API_KEY) {
      return res.status(500).json({
        error: { message: 'xKiro API key not configured', type: 'invalid_request_error', code: 500 },
      });
    }

    const { model, messages, temperature, max_tokens, stream, top_p, frequency_penalty, presence_penalty } = req.body;

    if (!model || !messages) {
      return res.status(400).json({
        error: { message: 'Missing required fields: model and messages are required', type: 'invalid_request_error', code: 400 },
      });
    }

    const xkiroModel = resolveModel(model);
    console.log(`Model: ${model} -> ${xkiroModel}${stream ? ' (stream)' : ''}`);

    const xkiroRequest = {
      model: xkiroModel,
      messages,
      temperature: temperature !== undefined ? temperature : 0.6,
      max_tokens: max_tokens || DEFAULT_MAX_TOKENS,
      stream: stream || false,
    };
    if (top_p !== undefined) xkiroRequest.top_p = top_p;
    if (frequency_penalty !== undefined) xkiroRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty !== undefined) xkiroRequest.presence_penalty = presence_penalty;

    const response = await callXkiro(xkiroRequest, !!stream);

    if (response.status >= 400) {
      let errData = response.data;
      if (stream && errData && typeof errData.on === 'function') {
        errData = await new Promise((resolve) => {
          let buf = '';
          errData.on('data', (c) => (buf += c.toString()));
          errData.on('end', () => {
            try { resolve(JSON.parse(buf)); } catch { resolve({ error: { message: buf } }); }
          });
          errData.on('error', () => resolve(null));
        });
      }
      console.error('xKiro API error:', response.status, errData);
      const headers = {};
      if (response.headers['retry-after']) headers['retry-after'] = response.headers['retry-after'];
      return res.set(headers).status(response.status).json({
        error: {
          message: errData?.error?.message || errData?.detail || 'xKiro API request failed',
          type: 'invalid_request_error',
          code: response.status,
        },
      });
    }

    // ---- STREAM ----
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) {
            res.write(line + '\n\n');
            continue;
          }
          try {
            const data = JSON.parse(line.slice(6));
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            console.error('Parse stream chunk error:', e.message);
            res.write(line + '\n\n');
          }
        }
      });

      response.data.on('end', () => { res.end(); });
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        if (!res.writableEnded) { res.write('data: [DONE]\n\n'); res.end(); }
      });
      req.on('close', () => {
        if (response.data && typeof response.data.destroy === 'function') response.data.destroy();
      });
      return;
    }

    // ---- NON-STREAM ----
    if (!Array.isArray(response.data?.choices)) {
      console.error('Réponse xKiro inattendue:', response.data);
      return res.status(502).json({
        error: { message: 'Réponse inattendue de xKiro', type: 'invalid_request_error', code: 502 },
      });
    }

    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: response.data.choices.map((choice) => ({
        index: choice.index,
        message: { role: choice.message?.role || 'assistant', content: choice.message?.content || '' },
        finish_reason: choice.finish_reason,
      })),
      usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };

    res.json(openaiResponse);
  } catch (error) {
    console.error('Proxy error:', error.message);
    const status = error.response?.status || (error.code === 'ECONNABORTED' ? 504 : 500);
    res.status(status).json({
      error: {
        message: error.code === 'ECONNABORTED' ? 'Timeout en attendant xKiro' : error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: status,
      },
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.method} ${req.path} not found`, type: 'invalid_request_error', code: 404 },
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log('========================================');
  console.log(`OpenAI to xKiro Proxy on port ${PORT}`);
  console.log(`Rate limit interne : ${RATE_LIMIT_RPM} req/min`);
  console.log(`xKiro API Key: ${XKIRO_API_KEY ? 'YES' : 'NO'}`);
  console.log('========================================');
});

module.exports = app;
