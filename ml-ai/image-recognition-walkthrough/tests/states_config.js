// The photos the tests run through: the page's default photo, and a mix of right and wrong answers.
const defaultPhoto = D => D.photos.filter(p => p.split === 'test').find(p => p.label === 'malamute' && p.pb > 0.8);
function photos(D) {
  const test = D.photos.filter(p => p.split === 'test');
  const wrongB = test.filter(p => (p.pb > 0.5 ? 'malamute' : 'husky') !== p.label);
  const pick = [defaultPhoto(D), test.find(p => p.label === 'husky'), ...wrongB.slice(0, 2), test[test.length - 1]];
  return [...new Set(pick.filter(Boolean))];
}
module.exports = { defaultPhoto, photos };
