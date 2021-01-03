import XCTest
@testable import ArticleClassifierCore

final class TfIdfAdvancedTests: XCTestCase {
    
    private let text = "ios 10 was the best ios. ios 13 is even better.".lowercased()
    private let text2 = "ios 13 is best on the iphone xr!".lowercased()

    /// Test the tokenization
    func testTokenization() {

        self.measure {
            let tfIdf = TfIdf(texts: [text, text2])

            let textTerms = tfIdf.lemmasIn(text: text)
            let expectedTextTerms = Set(["be", "ios 10", "ios 13", "good", "better", "even", "the", "ios"])
            XCTAssertEqual(expectedTextTerms, textTerms)

            let textTermsVector = tfIdf.lemmaVector(from: text)
            let expectedtextTermsVector = ["be", "better", "even", "good", "ios", "ios 10", "ios 13",  "the"]
            XCTAssertEqual(expectedtextTermsVector, Array(textTermsVector))

            let tfIdfAllTermsVector = tfIdf.allTermsVector
            let tfIdfExpectedAllTermsVector = ["be", "better", "even", "good", "ios", "ios 10", "ios 13", "iphonexr", "on", "the"]
            XCTAssertEqual(tfIdfExpectedAllTermsVector, Array(tfIdfAllTermsVector))

            let tfIdfAllTerms = tfIdf.allTerms
            let tfIdfAllTermsExpected = Set(["be", "better", "even", "good", "ios", "ios 10", "ios 13", "iphonexr", "on", "the"])
            XCTAssertEqual(tfIdfAllTermsExpected, tfIdfAllTerms)
        }
    }

    /// Test the frequencies
    func testFrequency() {
        let tfIdf = TfIdf(texts: [text, text2])

        let termDocumentFrequency = tfIdf.termsDocumentFrequency
        let termDocumentFrequencyExpected = ["be" : 2, "better": 1, "even": 1,
                                             "good": 2, "ios": 1, "ios 10": 1,
                                             "ios 13": 2, "iphonexr": 1, "on": 1,
                                             "the": 2]
        XCTAssertEqual(termDocumentFrequencyExpected, termDocumentFrequency)

        let termFrequencyVector = tfIdf.termFrequencyVector(text: text)
        let termFrequencyVectorExpected = [2, // be
                                           1, // better
                                           1, // even
                                           1, // good
                                           1, // ios
                                           1, // ios 10
                                           1, // ios 13
                                           0, // iphonexr
                                           0, // on
                                           1 // the
        ]
        XCTAssertEqual(termFrequencyVectorExpected, Array(termFrequencyVector))

        let idfTermsToNumTextContainingTerm = tfIdf.invertedDocumentFrequencyVector()
        let idfTermsToNumTextContainingTermExpected: [Double]  = [
            log(2/2), // be
            log(2/1), // better
            log(2/1), // even
            log(2/2), // good
            log(2/1), // ios
            log(2/1), // ios 10
            log(2/2), // ios 13
            log(2/1), // iphonexr
            log(2/1), // on
            log(2/2) // the
        ]
        XCTAssertEqual(idfTermsToNumTextContainingTermExpected, Array(idfTermsToNumTextContainingTerm))
    }
}
