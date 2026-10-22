import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { distance, evaluate, FaqSearch, misspell, parseWordList, Speller, StaticModel, writeWordList } from '../dist/node.js';

const json = (f) => JSON.parse(readFileSync(new URL(`../examples/${f}`, import.meta.url), 'utf8'));
const entries = json('store-faq.json');
const tests = json('store-tests.json');
const model = await StaticModel.load();
const faq = await FaqSearch.create(entries, { model });

test('answers questions asked in other words', () => {
  for (const [q, id] of [
    ['when will my stuff arrive', 'shipping-time'],
    ['do you take paypal', 'payment-methods'],
    ['the mug came smashed', 'damaged'],
    ['stop sending me newsletters', 'unsubscribe'],
  ]) {
    const r = faq.ask(q);
    assert.equal(r.answer?.entry.id, id, q);
    assert.equal(r.matches.length, 3);
    assert.ok(r.matches[0].score >= r.matches[1].score);
  }
});

test('turns away questions the FAQ does not answer, but still lists the closest', () => {
  const r = faq.ask('how tall is mount everest');
  assert.equal(r.answer, null);
  assert.equal(r.matches.length, 3);
});

test('corrects spelling and says so', () => {
  const r = faq.ask('how do I chek my gift crad balance');
  assert.equal(r.corrected, 'how do I check my gift card balance');
  assert.equal(r.answer?.entry.id, 'gift-card-balance');
  assert.equal(faq.ask('how do I check my gift card balance').corrected, null);
});

test('accuracy on the example tests stays where it was measured', () => {
  const r = evaluate(faq, tests);
  assert.ok(r.top1 >= 0.62 && r.top3 >= 0.86, JSON.stringify(r));
  assert.equal(r.offTopicTurnedAway, 1);
  const typos = evaluate(faq, tests, { typos: 2 });
  assert.ok(typos.top1 >= 0.58 && typos.top3 >= 0.82, JSON.stringify(typos));
});

test('filters, adds and removes entries', async () => {
  const f = new FaqSearch(model, entries.slice(0, 10), { words: false, threshold: 0.3 });
  assert.equal(f.entries.length, 10);
  f.add({ id: 'loyalty', q: 'Do you have a loyalty programme?', a: 'Yes: one point per dollar.' });
  assert.equal(f.ask('do I earn points when I shop').answer?.entry.id, 'loyalty');
  assert.notEqual(f.ask('do I earn points when I shop', { filter: (e) => e.id !== 'loyalty' }).matches[0].entry.id, 'loyalty');
  f.remove((e) => e.id === 'loyalty');
  assert.equal(f.entries.length, 10);
});

test('alt phrasings fix a miss', () => {
  const base = { id: 'card-types', q: 'What types of credit cards do you accept?', a: 'Visa and Mastercard.' };
  const other = { id: 'hours', q: 'What are your opening hours?', a: 'Nine to five.' };
  const without = new FaqSearch(model, [base, other], { words: false });
  const withAlt = new FaqSearch(model, [{ ...base, alt: ['do you take amex'] }, other], { words: false });
  assert.ok(withAlt.ask('do you take amex').matches[0].score > without.ask('do you take amex').matches[0].score + 0.2);
});

test('a fixed threshold is used as given; auto calibrates per FAQ', () => {
  assert.equal(new FaqSearch(model, entries, { threshold: 0.5 }).threshold, 0.5);
  const auto = faq.threshold;
  assert.ok(auto > 0.25 && auto < 0.5, String(auto));
});

test('speller: leaves known and short words, fixes typos, prefers the FAQ', () => {
  const s = new Speller(model, { preferred: ['Reconcile your remittance'], known: ['reassign'] });
  assert.equal(s.correct('can reprots exprot to csv'), 'can reports export to csv');
  assert.equal(s.correct('reconcle the remitance'), 'reconcile the remittance');
  assert.equal(s.correct('reassign it'), 'reassign it');
  assert.equal(s.correct('teh cat'), 'teh cat'); // under 4 letters
  assert.equal(s.correct('Café'), 'Café');
  assert.equal(s.correct('whens my stuf arriving'), 'when my stuff arriving'); // the commoner of stuff/stu/stud
});

test('edit distance counts a swap as one edit', () => {
  assert.equal(distance('form', 'from'), 1);
  assert.equal(distance('kitten', 'sitting'), 3);
  assert.equal(distance('kitten', 'sitting', 2), Infinity);
});

test('word lists round-trip front-coded', () => {
  const words = ['apple', 'applesauce', 'apply', 'banana', 'band'];
  const text = writeWordList(words);
  assert.ok(text.startsWith('#front-coded\n'));
  assert.deepEqual(parseWordList(text), words);
  assert.deepEqual(parseWordList('one\ntwo\n'), ['one', 'two']);
});

test('misspell is repeatable and changes the asked number of words', () => {
  const q = 'where can I download the invoice for my order';
  assert.equal(misspell(q, 2, 7), misspell(q, 2, 7));
  const changed = misspell(q, 2, 7).split(' ').filter((w, i) => w !== q.split(' ')[i]).length;
  assert.ok(changed >= 1 && changed <= 2);
});
