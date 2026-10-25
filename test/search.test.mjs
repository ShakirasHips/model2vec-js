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
    ['when will my stuff arrive', 'shipping-time'],
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
  assert.ok(top1 / answerable.length >= 0.58 && top3 / answerable.length >= 0.82, `${top1} ${top3} of ${answerable.length}`);
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
