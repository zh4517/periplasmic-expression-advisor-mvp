const presets = {
  antibody: `>Antibody Fragment A | sp=pelB
MKYLLPTAAAGLLLLAAQPAMADIQMTQSPSSLSASVGDRVTITCRASQGISSALAWYQQKPGKAPKLLIY
>Antibody Fragment B | sp=pelB
MKYLLPTAAAGLLLLAAQPAMAQVQLVQSGAEVKKPGASVKVSCKASGYTFTDYYMHWVKQ
>Antibody Fragment C | sp=pelB
MKYLLPTAAAGLLLLAAQPAMADIVMTQSPSSLSASVGDRVTITCRASQGISSYLNWYQQKPGKAPKLLIY`,
  enzyme: `>Oxidoreductase Variant 01 | sp=ompA
MKKTAIAIAVALAGFATVAQAAEADKIRKYLNQEGIDVIEVAPNQATKPKGVVFAEGDPK
>Hydrolase Variant 02 | sp=pelB
MKYLLPTAAAGLLLLAAQPAMADKVTLELGEKVGKLVAVPAGSNVVAMTDPDGTYH
>Ligase Variant 03 | sp=ompA
MKKTAIAIAVALAGFATVAQAAELKAVEEAGVDVVVVDDTPRHYRVDLAAEAGADVV`,
  mixed: `>Reporter Protein C | sp=pelB
MKYLLPTAAAGLLLLAAQPAMASSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATNGKLTLK
>Binding Protein D | sp=pelB
MKYLLPTAAAGLLLLAAQPAMAQVQLVQSGAEVKKPGASVKVSCKASGYTFTDYYMHWVKQ
>Enzyme Variant E | sp=ompA
MKKTAIAIAVALAGFATVAQAAEADKIRKYLNQEGIDVIEVAPNQATKPKGVVFAE
>Novel Scaffold F | sp=pelB
MKYLLPTAAAGLLLLAAQPAMAGQSVLSLQPGQSLRLSCAASGFTFSSYAMSWVRQAPGKGLEW`,
  boundary: `>Known-like Construct | sp=pelB
MKYLLPTAAAGLLLLAAQPAMADIQMTQSPSSLSASVGDRVTITCRASQGISSALAWYQQKPGKAPKLLIY
>Nonstandard Residue Example | sp=pelB
MKYLLPTAAAGLLLLAAQPAMAUCGELFTGVVPILVELDGDVNGHKFS
>Very Short Chain | sp=pelB
MKYLLPTAAAGLLLLAAQPAMAGK
>Unknown Signal Peptide | sp=customSP
MNNNKLLAALLAAGFAAAQAADKVTLELGEKVGKLVAVPAGSNVVAMTDPDGTYH`
};

const examples = Object.values(presets);
let exampleIndex = 0;
let activePreset = 'antibody';
const form = document.querySelector('#analysis-form');
const fasta = document.querySelector('#fasta');
const empty = document.querySelector('#result-empty');
const content = document.querySelector('#result-content');
const list = document.querySelector('#result-list');
const recommendControl = document.querySelector('#recommend-control');
const subtitle = document.querySelector('#results-subtitle');
const standardResidues = new Set('ACDEFGHIKLMNPQRSTVWY');

function parseRecords(text) {
  // 读取 FASTA 文本。每个 > 标题与其后若干序列行合并为一个候选记录。
  const records = [];
  let name = '';
  let sequence = '';
  const flush = () => {
    if (name) records.push({name: name.replace(/\|\s*sp=.*$/i, '').trim(), sequence: sequence.replace(/\s/g, '').toUpperCase()});
  };
  text.split(/\r?\n/).forEach(line => {
    if (line.startsWith('>')) { flush(); name = line.slice(1); sequence = ''; }
    else { sequence += line.trim(); }
  });
  flush();
  return records.length ? records : [{name: '候选构建体 A', sequence: text.replace(/\s/g, '').toUpperCase()}];
}

function makeDemoCandidates(sourceText, count) {
  // 根据当前预设扩展出多条“演示变体”，只用于展示批量排序交互。
  // 只从标准氨基酸序列生成变体，避免把边界案例误生成为正常候选。
  const bases = parseRecords(sourceText).filter(record => [...record.sequence].every(letter => standardResidues.has(letter)));
  const alphabet = 'ACDEFGHIKLMNPQRSTVWY';
  const filler = 'GASGSGSTGAPGATGAGAGATGSGS';
  return Array.from({length: count}, (_, index) => {
    const base = bases[index % bases.length].sequence;
    let sequence = base;
    // 补齐长度，确保演示变体落在真实模型的 80 aa 最低长度附近。
    while (sequence.length < 96) sequence += filler;
    const chars = sequence.slice(0, 150).split('');
    const mutationCount = 1 + (index % 4);
    for (let step = 0; step < mutationCount; step += 1) {
      const position = 25 + ((index * 17 + step * 29) % (chars.length - 25));
      chars[position] = alphabet[(index * 7 + step * 11) % alphabet.length];
    }
    const signal = /sp=ompA/i.test(sourceText) && index % 2 ? 'ompA' : 'pelB';
    return `>演示候选变体 ${String(index + 1).padStart(2, '0')} | sp=${signal}\n${chars.join('')}`;
  }).join('\n');
}

function scoreRecord(record) {
  // 前端演示用的确定性评分规则，不等同于 advisor.py 的真实 Python 模型。
  const {name, sequence} = record;
  // 先拦截非标准残基；真实特征提取也无法处理这些字符。
  const invalid = [...sequence].filter(letter => !standardResidues.has(letter));
  if (invalid.length) return {name, invalid: `含非标准残基 ${[...new Set(invalid)].join('、')}`, score: -1};
  if (sequence.length < 35) return {name, invalid: `序列过短（${sequence.length} aa）`, score: -1};
  const fraction = letters => [...sequence].filter(letter => letters.includes(letter)).length / sequence.length;
  const hydrophobic = fraction('AVILMFWY');
  const charged = fraction('DEKR');
  const proline = fraction('P');
  const lengthFit = Math.max(0, 1 - Math.abs(sequence.length - 180) / 250);
  const signalBonus = /\|\s*sp=(pelB|ompa)/i.test(name) ? 7 : 0;
  let hash = 0;
  for (const letter of sequence) hash = (hash * 31 + letter.charCodeAt(0)) % 997;
  const motifVariation = (hash % 23) - 11;
  // 用长度、组成和已知信号肽构造演示分数，使不同输入展示不同排序。
  const score = Math.round(38 + lengthFit * 23 + hydrophobic * 46 - charged * 19 - proline * 26 + signalBonus + motifVariation);
  const yieldMgl = Math.max(2, Math.round(4 + Math.exp(score / 26) * 6.8));
  return {name, score, yieldMgl, low: Math.max(0.5, yieldMgl / 4), high: yieldMgl * 4, length: sequence.length};
}

// 根据数量级压缩显示精度，避免结果卡片出现过多无意义小数。
function format(value) { return value >= 100 ? Math.round(value) : value >= 10 ? value.toFixed(1).replace(/\.0$/, '') : value.toFixed(1); }

function renderResults(records) {
  // 先按演示得分排序，再生成候选列表和“首批实验数量”选择器。
  // 无法预测的候选保留在列表末尾，用原因替代虚假的排名。
  const valid = records.filter(item => item.score >= 0).sort((a, b) => b.score - a.score);
  const invalid = records.filter(item => item.score < 0);
  const items = [...valid, ...invalid];
  const defaultCount = Math.min(3, valid.length);
  list.innerHTML = items.map((item, index) => {
    if (item.invalid) return `<article class="candidate invalid"><span class="rank-num">—</span><div><div class="candidate-name">${item.name}</div><div class="candidate-meta">无法生成预测：${item.invalid}</div></div><div class="yield"><strong>不排序</strong><span>超出模型输入范围</span></div></article>`;
    return `<article class="candidate ${index < defaultCount ? 'priority' : ''}" data-rank="${index + 1}"><span class="rank-num">0${index + 1}</span><div><div class="candidate-name">${item.name}</div><div class="candidate-meta">${item.length} aa</div></div><div class="yield"><strong>${format(item.yieldMgl)} mg/L</strong><span>区间 ${format(item.low)} – ${format(item.high)} mg/L</span></div></article>`;
  }).join('');
  if (!valid.length) {
    recommendControl.textContent = '没有符合输入要求、可进入首批实验的候选构建体';
    return;
  }
  recommendControl.innerHTML = `推荐进入首批实验 <label>数量 <select id="recommend-count">${Array.from({length: valid.length}, (_, index) => `<option value="${index + 1}" ${index + 1 === defaultCount ? 'selected' : ''}>${index + 1}</option>`).join('')}</select></label>`;
  // 用户改变 n 时，仅更新高亮范围，不重新计算排序。
  recommendControl.querySelector('#recommend-count').addEventListener('change', event => {
    const selected = Number(event.target.value);
    list.querySelectorAll('.candidate[data-rank]').forEach(card => {
      card.classList.toggle('priority', Number(card.dataset.rank) <= selected);
    });
  });
}

form.addEventListener('submit', event => {
  event.preventDefault();
  const button = form.querySelector('.analyze');
  button.innerHTML = '正在评估…';
  button.disabled = true;
  subtitle.textContent = '正在进行 20 次重采样演示评估';
  // 这里的短延迟仅模拟后台计算过程，便于演示交互状态。
  setTimeout(() => {
    const records = parseRecords(fasta.value).map(scoreRecord);
    renderResults(records);
    empty.classList.add('hidden');
    content.classList.remove('hidden');
    subtitle.textContent = `已完成 ${records.length} 个候选构建体的保守排序`;
    button.innerHTML = '重新分析 <span>→</span>';
    button.disabled = false;
  }, 700);
});

document.querySelector('#sample-button').addEventListener('click', () => {
  exampleIndex = (exampleIndex + 1) % examples.length;
  fasta.value = examples[exampleIndex];
  activePreset = Object.keys(presets)[exampleIndex];
  document.querySelectorAll('.preset').forEach(button => button.classList.remove('active'));
});

document.querySelector('#generate-button').addEventListener('click', () => {
  // 生成指定数量的演示变体，常用于演示 Top-10/首批 n 个的工作流。
  const count = Number(document.querySelector('#candidate-count').value);
  fasta.value = makeDemoCandidates(presets[activePreset], count);
});

document.querySelectorAll('.preset').forEach(button => {
  button.addEventListener('click', () => {
    fasta.value = presets[button.dataset.preset];
    activePreset = button.dataset.preset;
    document.querySelectorAll('.preset').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
  });
});
