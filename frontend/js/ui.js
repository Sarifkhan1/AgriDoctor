/**
 * AgriDoctor AI - Safe result rendering.
 *
 * SECURITY: never interpolate untrusted (user/model) strings into innerHTML.
 * Text goes through `esc()` (or is set via textContent); only our own markup
 * templates use innerHTML, and only on already-escaped content (`mdSafe`).
 */

const UI = (() => {
  // --- escaping helpers ---
  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }
  // Minimal markdown: escape FIRST, then apply bold/italic on the escaped text.
  function mdSafe(s) {
    return esc(s)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>');
  }

  // --- tiny DOM builder ---
  // opts: { cls, text, html (pre-sanitized), attrs, on }
  function el(tag, opts = {}, children = []) {
    const node = document.createElement(tag);
    if (opts.cls) node.className = opts.cls;
    if (opts.text != null) node.textContent = opts.text;
    if (opts.html != null) node.innerHTML = opts.html; // caller must sanitize
    if (opts.attrs) for (const [k, v] of Object.entries(opts.attrs)) node.setAttribute(k, v);
    if (opts.on) for (const [k, v] of Object.entries(opts.on)) node.addEventListener(k, v);
    for (const c of [].concat(children)) if (c) node.appendChild(c);
    return node;
  }

  const CROP_EMOJI = {
    tomato: '🍅', potato: '🥔', rice: '🌾', maize: '🌽', chili: '🌶️', cucumber: '🥒',
    pepper: '🫑', eggplant: '🍆',
    cattle: '🐄', goat: '🐐', sheep: '🐑', poultry: '🐔',
  };
  const SUPPORTED = [
    'Tomato 🍅', 'Potato 🥔', 'Rice 🌾', 'Maize 🌽', 'Chili 🌶️', 'Cucumber 🥒',
    'Pepper 🫑', 'Eggplant 🍆',
    'Cattle 🐄', 'Goat 🐐', 'Sheep 🐑', 'Poultry 🐔',
  ];

  function titleCase(s) {
    return (s || '').replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function diseaseTitle(r) {
    if (r.disease_name) return `${titleCase(r.detected_crop || '')} — ${r.disease_name}`.trim();
    return titleCase(r.detected_crop || 'Result');
  }

  // --- meter component ---
  function meter(icon, label, pct, tone) {
    const bar = el('div', { cls: 'meter__track' }, [
      el('div', { cls: `meter__fill meter__fill--${tone}`, attrs: { style: `width:${pct}%` } }),
    ]);
    return el('div', { cls: 'meter' }, [
      el('div', { cls: 'meter__head' }, [
        el('span', { cls: 'meter__label', text: `${icon} ${label}` }),
        el('span', { cls: 'meter__value', text: `${pct}%` }),
      ]),
      bar,
    ]);
  }

  function bulletList(cls, items, bullet) {
    const ul = el('ul', { cls });
    (items || []).forEach((it) => {
      ul.appendChild(el('li', { html: `<span class="bullet">${bullet}</span> ${mdSafe(it)}` }));
    });
    return ul;
  }

  // --- generic state card (rejection / info) ---
  function stateCard(emoji, title, message, tone = 'info', extra = null) {
    return el('div', { cls: `state-card state-card--${tone}` }, [
      el('div', { cls: 'state-card__icon', text: emoji }),
      el('h2', { cls: 'state-card__title', text: title }),
      el('p', { cls: 'state-card__msg', html: mdSafe(message || '') }),
      extra,
    ]);
  }

  function supportedChips() {
    const wrap = el('div', { cls: 'chips' });
    SUPPORTED.forEach((c) => wrap.appendChild(el('span', { cls: 'chip', text: c })));
    return wrap;
  }

  // --- full diagnosis / healthy card ---
  function diagnosisCard(r, ctx) {
    const urgency = r.urgency_level || 'medium';
    const conf = Math.round((r.confidence || 0) * 100);
    const sev = Math.round((r.severity_score || 0) * 100);
    const sevTone = sev < 33 ? 'low' : sev < 66 ? 'medium' : 'high';
    const isHealthy = r.kind === 'healthy';

    // Left column: image + quick facts
    const left = el('div', { cls: 'diag__left' });
    if (ctx && ctx.imageUrl) {
      left.appendChild(
        el('div', { cls: 'diag__imgwrap' }, [
          el('img', { cls: 'diag__img', attrs: { src: ctx.imageUrl, alt: 'Uploaded leaf' } }),
        ])
      );
    }
    const facts = el('div', { cls: 'diag__facts' });
    facts.appendChild(fact('Crop', `${CROP_EMOJI[(r.detected_crop || '').toLowerCase()] || '🌱'} ${titleCase(r.detected_crop || '—')}`));
    if (r.category) facts.appendChild(fact('Type', titleCase(r.category)));
    if (ctx && ctx.onsetText) facts.appendChild(fact('Onset', ctx.onsetText));
    if (ctx && ctx.spread) facts.appendChild(fact('Spread', titleCase(ctx.spread)));
    left.appendChild(facts);
    left.appendChild(
      el('div', { cls: 'diag__meters' }, [
        meter('🎯', 'AI Confidence', conf, conf < 45 ? 'medium' : 'low'),
        meter('🌡️', 'Infection Level', sev, sevTone),
      ])
    );

    // Right column: diagnosis + advice
    const right = el('div', { cls: 'diag__right' });
    right.appendChild(
      el('div', { cls: `diag__header diag__header--${isHealthy ? 'healthy' : urgency}` }, [
        el('span', { cls: 'diag__badge', text: isHealthy ? 'Healthy' : `${titleCase(urgency)} priority` }),
        el('h2', { cls: 'diag__title', text: diseaseTitle(r) }),
        el('div', { cls: 'diag__conf' }, [
          el('span', { text: `${isHealthy ? 'Looks healthy' : 'AI match'} · ${conf}% confidence` }),
          el('span', {
            cls: `diag__provider ${r.provider === 'local_cnn' ? 'diag__provider--local' : 'diag__provider--cloud'}`,
            text: r.provider === 'local_cnn' ? '🧠 Local AI' : '☁️ Cloud AI'
          }),
        ]),
      ])
    );

    if (r.matches_hint === false && r.detected_crop) {
      right.appendChild(
        stateBanner(`Heads up: you selected a different crop, but this looks like a ${esc(titleCase(r.detected_crop))} leaf — diagnosed accordingly.`)
      );
    }

    if (r.confidence != null && r.confidence < 0.45 && !isHealthy) {
      right.appendChild(stateBanner('Confidence is low — treat this as a possibility and consider a clearer photo or a second opinion.'));
    }

    if (r.visual_evidence) {
      right.appendChild(section('🔬', 'What the AI saw', el('p', { cls: 'section__text', html: mdSafe(r.visual_evidence) })));
    }

    const advice = r.advice || {};
    if (advice.summary) {
      right.appendChild(section('📜', 'Diagnostic Summary', el('p', { cls: 'section__text', html: mdSafe(advice.summary) })));
    }
    if ((advice.what_to_do_now || []).length) {
      const steps = el('div', { cls: 'steps' });
      advice.what_to_do_now.forEach((it, i) => {
        steps.appendChild(
          el('div', { cls: 'step-row' }, [
            el('div', { cls: 'step-row__num', text: String(i + 1) }),
            el('div', { cls: 'step-row__body', html: mdSafe(it) }),
          ])
        );
      });
      right.appendChild(section('⚡', 'Immediate Action Plan', steps));
    }
    if ((advice.prevention || []).length) {
      right.appendChild(section('🛡️', 'Long-term Prevention', bulletList('clean-list', advice.prevention, '✨')));
    }
    if ((advice.when_to_get_help || []).length) {
      right.appendChild(section('🚨', 'When to Escalate', bulletList('clean-list', advice.when_to_get_help, '🔺')));
    }

    right.appendChild(
      el('div', { cls: 'safety' }, [
        el('div', { cls: 'safety__icon', text: 'ℹ️' }),
        el('div', {}, [
          el('strong', { text: 'Expert consultation recommended' }),
          el('p', { cls: 'safety__text', html: mdSafe(r.safety_note || '') }),
        ]),
      ])
    );

    return el('div', { cls: 'diag' }, [left, right]);
  }

  function fact(label, value) {
    return el('div', { cls: 'fact' }, [
      el('span', { cls: 'fact__k', text: label }),
      el('span', { cls: 'fact__v', text: value }),
    ]);
  }
  function section(icon, title, body) {
    return el('div', { cls: 'section' }, [
      el('div', { cls: 'section__icon', text: icon }),
      el('div', { cls: 'section__content' }, [el('h3', { text: title }), body]),
    ]);
  }
  function stateBanner(html) {
    return el('div', { cls: 'inline-banner', html: `⚠️ ${html}` });
  }

  // --- public: render a result into a container ---
  function renderResult(container, r, ctx) {
    container.innerHTML = '';
    let node;
    switch (r.kind) {
      case 'not_a_leaf':
        node = stateCard('🌱', "That's not a plant leaf",
          r.message || 'Please photograph a single affected leaf, filling most of the frame.', 'warn');
        break;
      case 'unsupported_crop':
        node = stateCard('🚫', 'Crop not supported yet',
          r.message || `This looks like ${esc(titleCase(r.detected_crop || 'another crop'))}.`,
          'warn', supportedChips());
        break;
      case 'low_confidence':
        node = stateCard('🔍', 'Image needs to be clearer',
          r.message || 'Retake a close, well-lit photo of a single affected leaf.', 'info');
        break;
      case 'healthy':
      case 'diagnosis':
        node = diagnosisCard(r, ctx || {});
        break;
      default:
        node = stateCard('⚠️', 'Something went wrong', 'Unexpected response. Please try again.', 'warn');
    }
    container.appendChild(node);
  }

  function renderError(container, message) {
    container.innerHTML = '';
    container.appendChild(
      stateCard('⚠️', 'Analysis failed', message || 'Please try again in a moment.', 'warn')
    );
  }

  return { esc, mdSafe, renderResult, renderError, CROP_EMOJI, titleCase };
})();

window.UI = UI;
