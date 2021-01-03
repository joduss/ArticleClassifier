import Cocoa
import CoreML
import NaturalLanguage



let tagger = NLTagger(tagSchemes: [.lemma, .nameTypeOrLexicalClass])

var text = "L'iphone qu'Apple a présenté hier ne sera disponible qu'au moins de septembre 2020."

var processedText = ""


tagger.string = text

let gazetter = try NLGazetteer(dictionary: ["smartphone": ["iphone", "iphone xr", "iphone 12 pro"], "iphone": ["iphone"]], language: .french)
tagger.setGazetteers([gazetter], for: .nameTypeOrLexicalClass)

// Remove determinant
tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameTypeOrLexicalClass, options: [.omitWhitespace, .omitPunctuation, .joinNames, .omitOther], using: {
    tag, range in
    
    let term = String(text[range])
    
    if tag == NLTag.preposition {
        print("\(term) => ❌ (prep)")
    }
    else if tag == NLTag.adverb {
        print("\(term) => ❌ (adv)")
    }
    else if tag == NLTag.conjunction {
        print("\(term) => ❌ (conj)")
    }
    else if tag == NLTag.determiner {
        print("\(term) => ❌ (det)")
    }
    else {
        print("\(term) => OK \(tag!.rawValue)")
        processedText += " \(term)"
    }
    return true
})

print()
// Remove determinant
tagger.string = processedText
var finalText = ""

tagger.enumerateTags(in: processedText.startIndex..<processedText.endIndex, unit: .word, scheme: .lemma, options: [.omitWhitespace, .omitPunctuation, .joinNames], using: {
    tag, range in

    let term = String(processedText[range])

    guard let detectedtag = tag else {
        print("tag not found for \(term)")
        return true
    }

    guard let lemmatizedWord = tag?.rawValue else {
        print("Term: \(term)")
        return true
    }

    print("\(term) => \(detectedtag.rawValue)")

    finalText += " \(lemmatizedWord)"

    return true
})

print()
print(finalText.trimmingCharacters(in: .whitespacesAndNewlines))





