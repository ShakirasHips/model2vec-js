import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import MiniSemanticSearch, { StaticModel } from '../dist/node.js';

const json = (f) => JSON.parse(readFileSync(new URL(`../examples/${f}`, import.meta.url), 'utf8'));
const docs = json('store-faq.json');
const tests = json('store-tests.json');
const model = await StaticModel.load();
const make = (options = {}) => new MiniSemanticSearch({ model, fields: ['section', 'q', 'a'], storeFields: ['q'], ...options });
const search = make({ searchOptions: { boost: { q: 3 } } });
search.addAll(docs);

test('finds documents asked about in other words', () => {
  for (const [q, id] of [
    ['how long does delivery take', 'shipping-time'],
    ['do you take paypal', 'payment-methods'],
    ['the mug came smashed', 'damaged'],
    ['stop sending me newsletters', 'unsubscribe'],
  ]) {
    const results = search.search(q);
    assert.equal(results[0].id, id, q);
    assert.equal(results.length, docs.length);
    assert.ok(results.every((r, i) => !i || results[i - 1].score >= r.score));
  }
});

test('results carry the id, score and stored fields only', () => {
  const [r] = search.search('do you take paypal');
  assert.deepEqual(Object.keys(r).sort(), ['id', 'q', 'score']);
  assert.equal(r.q, 'What payment methods do you accept?');
  assert.deepEqual(search.getStoredFields('payment-methods'), { q: 'What payment methods do you accept?' });
  assert.equal(search.getStoredFields('nope'), undefined);
});

test('accuracy on the example tests stays where it was measured', () => {
  let top1 = 0, top3 = 0;
  const answerable = tests.filter((t) => t.ids.length);
  for (const t of answerable) {
    const ids = search.search(t.q).slice(0, 3).map((r) => r.id);
    if (t.ids.includes(ids[0])) top1++;
    if (ids.some((id) => t.ids.includes(id))) top3++;
  }
  assert.ok(top1 / answerable.length >= 0.63 && top3 / answerable.length >= 0.84, `${top1} ${top3} of ${answerable.length}`);
});

test('search options: filter, minScore, fields and boost', () => {
  const q = 'do you take paypal';
  assert.ok(search.search(q, { filter: (r) => r.id !== 'payment-methods' }).every((r) => r.id !== 'payment-methods'));
  const some = search.search(q, { minScore: 0.4 });
  assert.ok(some.length > 0 && some.length < docs.length && some.every((r) => r.score >= 0.4));
  const sections = search.search('shipping', { fields: ['section'] });
  assert.ok(sections.slice(0, 4).every((r) => docs.find((d) => d.id === r.id).section === 'Shipping'));
  const plain = search.search(q, { boost: {} })[0].score;
  assert.notEqual(plain, search.search(q)[0].score); // the default boost is overridden
});

test('add, discard, remove, replace and removeAll, as in MiniSearch', () => {
  const s = make();
  s.addAll(docs.slice(0, 10));
  assert.equal(s.documentCount, 10);
  const loyalty = { id: 'loyalty', q: 'Do you have a loyalty programme?', a: 'Yes: one point per dollar.' };
  s.add(loyalty);
  assert.ok(s.has('loyalty'));
  assert.equal(s.search('do I earn points when I shop')[0].id, 'loyalty');
  assert.throws(() => s.add(loyalty), /duplicate ID loyalty/);
  assert.throws(() => s.add({ q: 'no id' }), /does not have ID field "id"/);
  s.replace({ ...loyalty, q: 'Is there a rewards scheme?' });
  assert.equal(s.getStoredFields('loyalty').q, 'Is there a rewards scheme?');
  s.remove(loyalty);
  assert.ok(!s.has('loyalty'));
  assert.throws(() => s.discard('loyalty'), /not in the index/);
  s.discardAll([docs[0].id, docs[1].id]);
  assert.equal(s.documentCount, 8);
  s.removeAll();
  assert.equal(s.documentCount, 0);
  assert.deepEqual(s.search('anything'), []);
});

test('scores stay right through adds, removals and changes of search options', () => {
  const fresh = (list, options) => {
    const s = make();
    s.addAll(list);
    return new Map(s.search('where is my refund', options).map((r) => [r.id, r.score]));
  };
  const s = make();
  s.addAll(docs.slice(0, 20));
  for (const options of [{}, { fields: ['q'] }, { boost: { a: 2 } }, {}]) {
    s.search('warm up', options);
    s.search('warm up', options); // options used twice running get a score matrix, which the changes below update
    s.addAll(docs.slice(20, 30));
    s.discardAll([docs[3].id, docs[25].id, docs[0].id]);
    s.replace({ ...docs[10], q: 'Where is my refund?' });
    const now = docs.slice(0, 30).filter((d) => ![docs[3].id, docs[25].id, docs[0].id].includes(d.id))
      .map((d) => (d.id === docs[10].id ? { ...d, q: 'Where is my refund?' } : d));
    const want = fresh(now, options), got = s.search('where is my refund', options);
    assert.equal(got.length, want.size);
    for (const r of got) assert.ok(Math.abs(r.score - want.get(r.id)) < 1e-6, `${r.id}`);
    assert.equal(got[0].id, [...want].sort((a, b) => b[1] - a[1])[0][0]);
    s.removeAll();
    s.addAll(docs.slice(0, 20));
  }
});

test('a first search with new options scores as later ones do', () => {
  const s = make();
  s.addAll([...docs, { id: 'alt', q: ['Can I pay by card?', 'Do you take amex?'], a: 'Yes.' }]);
  for (const options of [{}, { fields: ['q', 'a'] }, { boost: { q: 3, section: 0 } }]) {
    const first = s.search('do you take amex', options), again = s.search('do you take amex', options);
    assert.deepEqual(first.map((r) => r.id).slice(0, 5), again.map((r) => r.id).slice(0, 5));
    first.forEach((r, i) => assert.ok(Math.abs(r.score - again.find((a) => a.id === r.id).score) < 1e-6, `${r.id} ${i}`));
  }
});

test('addAllAsync, a custom idField and extractField', async () => {
  const s = make({ idField: 'key', fields: ['title'], extractField: (d, f) => (f === 'title' ? d.meta.title : d[f]) });
  await s.addAllAsync(docs.map((d) => ({ key: d.id, meta: { title: d.q } })), { chunkSize: 7 });
  assert.equal(s.documentCount, docs.length);
  assert.equal(s.search('do you take paypal')[0].id, 'payment-methods');
});

test('array fields count their closest item', () => {
  const base = { id: 'card-types', q: 'What types of credit cards do you accept?' };
  const other = { id: 'hours', q: 'What are your opening hours?' };
  const without = new MiniSemanticSearch({ model, fields: ['q', 'alt'] });
  without.addAll([base, other]);
  const withAlt = new MiniSemanticSearch({ model, fields: ['q', 'alt'] });
  withAlt.addAll([{ ...base, alt: ['how much is it', 'do you take amex'] }, other]);
  assert.ok(withAlt.search('do you take amex')[0].score > without.search('do you take amex')[0].score + 0.1);
});

test('create loads the bundled model; the constructor needs one', async () => {
  const s = await MiniSemanticSearch.create({ fields: ['q'] });
  s.add({ id: 1, q: 'How do I return an item?' });
  assert.equal(s.search('send it back')[0].id, 1);
  assert.throws(() => new MiniSemanticSearch({ fields: ['q'] }), /needs a loaded model/);
  assert.throws(() => new MiniSemanticSearch({ model, fields: [] }), /"fields"/);
});

test('keywordWeight: shared words count, from not at all to alone', () => {
  const s = new MiniSemanticSearch({ model, fields: ['text'] });
  s.addAll([
    { id: 'name', text: 'Form XK42 is for warehouse staff.' },
    { id: 'meaning', text: 'Tropical beach holidays by the warm sea.' },
    { id: 'other', text: 'How to file your tax return.' },
  ]);
  const q = 'seaside vacation xk42';
  assert.equal(s.search(q, { keywordWeight: 0 })[0].id, 'meaning');
  assert.equal(s.search(q)[0].id, 'name');
  const words = s.search(q, { keywordWeight: 1 });
  assert.equal(words[0].id, 'name');
  assert.ok(words[0].score > 0 && words[0].score <= 1);
  assert.equal(words[2].score, 0); // no word of the query is in it
  // a query with no words in it is scored on meaning alone
  assert.equal(s.search('?!').length, 3);
  // keyword scores keep to the fields searched
  const two = new MiniSemanticSearch({ model, fields: ['title', 'text'] });
  two.addAll([{ id: 1, title: 'Zanzibar', text: 'An island.' }, { id: 2, title: 'An island', text: 'Zanzibar.' }]);
  assert.equal(two.search('zanzibar', { fields: ['title'], keywordWeight: 1 })[0].id, 1);
  assert.equal(two.search('zanzibar', { fields: ['title'], keywordWeight: 1 })[1].score, 0);
  assert.equal(two.search('zanzibar', { fields: ['text'], keywordWeight: 1 })[0].id, 2);
});

test('long texts are split into passages, and the closest counts', () => {
  const filler = Array.from({ length: 12 }, (_, i) => `The committee met on day ${i} to review the quarterly budget figures.`).join(' ');
  const cats = 'Our cat likes to sleep on the warm windowsill. She naps there in the sunshine every afternoon. The kitten curls up beside her and dozes too.';
  const long = { id: 'long', text: `${filler} ${cats} ${filler}` };
  const other = { id: 'other', text: `${filler} ${filler}` };
  const scores = (options) => {
    const s = new MiniSemanticSearch({ model, fields: ['text'], passageLength: 40, searchOptions: { keywordWeight: 0 }, ...options });
    s.addAll([long, other]);
    return new Map(s.search('a kitten napping in the sun').map((r) => [r.id, r.score]));
  };
  const split = scores({}), whole = scores({ passageLength: 0 });
  assert.ok(split.get('long') > whole.get('long') + 0.2, `${split.get('long')} ${whole.get('long')}`);
  assert.ok(split.get('long') > split.get('other') + 0.2);
  // a text shorter than a passage scores the same either way
  const short = (options) => {
    const s = new MiniSemanticSearch({ model, fields: ['q'], ...options });
    s.addAll(docs);
    return s.search('do you take paypal').map((r) => r.score);
  };
  assert.deepEqual(short({}), short({ passageLength: 0 }));
});

test('scores stay right when most documents have been removed', () => {
  const many = Array.from({ length: 120 }, (_, i) => ({ ...docs[i % docs.length], id: `${i}` }));
  const s = make();
  s.addAll(many);
  s.search('where is my refund');
  s.search('where is my refund');
  s.discardAll(many.slice(0, 100).map((d) => d.id)); // enough for the keyword index to renumber what is left
  s.addAll(many.slice(0, 5));
  const fresh = make();
  fresh.addAll([...many.slice(100), ...many.slice(0, 5)]);
  const want = new Map(fresh.search('where is my refund').map((r) => [r.id, r.score]));
  const got = s.search('where is my refund');
  assert.equal(got.length, 25);
  for (const r of got) assert.ok(Math.abs(r.score - want.get(r.id)) < 1e-6, r.id);
});
