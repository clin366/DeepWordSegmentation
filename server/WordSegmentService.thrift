namespace py ai.botbrain.wordsegment
namespace java ai.botbrain.wordsegment

struct PosResult {
    1:string word,
    2:string posTag,
}

service WordSegmentService {
    bool alive(),
    list<string> segmentText(1:string input),
    list<list<string>> segmentTexts(1:list<string> inputs),
    list<PosResult> posTagging(1:list<string> words),
    list<list<PosResult>> posTaggings(1:list<list<string>> wordsList),
    list<PosResult> segmentWithPosTagging(1:string input),
    list<list<PosResult>> segmentWithPosTaggings(1:list<string> inputs)
}
