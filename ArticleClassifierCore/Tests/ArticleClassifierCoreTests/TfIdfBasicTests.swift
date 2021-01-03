//
//  TfIdfBasicTests.swift
//  ArticleClassifierCoreTests-macOS
//
//  Created by Jonathan Duss on 03.03.19.
//

import XCTest
@testable import ArticleClassifierCore

/// Test the TfIdf with normal words (no ios/mac specific terms
final class TfIdfBasicTests: XCTestCase {
    
    private let text = "This is a test method for a test!".lowercased()
    private let text2 = "I am another test!".lowercased()


    func testTokenization() {

        self.measure {
            let tfIdf = TfIdf(texts: [text, text2])

            let textTerms = tfIdf.lemmasIn(text: text)
            let expectedTextTerms = Set(["a", "for", "be", "method", "test", "this"])
            XCTAssertEqual(expectedTextTerms, textTerms)

            let textTermsVector = tfIdf.lemmaVector(from: text)
            let expectedtextTermsVector = ["a", "be", "for", "method", "test", "this"]
            XCTAssertEqual(expectedtextTermsVector, textTermsVector)

            let tfIdfAllTermsVector = tfIdf.allTermsVector
            let tfIdfExpectedAllTermsVector = ["a", "another", "be", "for", "i", "method", "test", "this"]
            XCTAssertEqual(tfIdfExpectedAllTermsVector, Array(tfIdfAllTermsVector))

            let tfIdfAllTerms = tfIdf.allTerms
            let tfIdfAllTermsExpected = Set(["a", "be", "another","for", "i", "method", "test", "this"])
            XCTAssertEqual(tfIdfAllTermsExpected, tfIdfAllTerms)
        }
    }

    /// Test the frequencies
    func testFrequency() {
        let tfIdf = TfIdf(texts: [text, text2])

        let termDocumentFrequency = tfIdf.termsDocumentFrequency
        let termDocumentFrequencyExpected = ["a": 1, "be" : 2, "another" : 1,
                                             "for" : 1, "i" : 1,
                                             "method" : 1, "test" : 2, "this" : 1]
        XCTAssertEqual(termDocumentFrequencyExpected, termDocumentFrequency)

        let termFrequencyVector = tfIdf.termFrequencyVector(text: text)
        let termFrequencyVectorExpected = [
            2, // a
            0, // another
            1, // be (is)
            1, // for
            0, // i
            1, // method
            2, // test
            1  // this
        ]
        XCTAssertEqual(termFrequencyVectorExpected, Array(termFrequencyVector))

        let idfTermsToNumTextContainingTerm = tfIdf.invertedDocumentFrequencyVector()
        let idfTermsToNumTextContainingTermExpected: [Double]  = [
            log(2/1), // a
            log(2/1), // another
            log(2/2), // be (am and is)
            log(2/1), // for
            log(2/1), // i
            log(2/1), // method
            log(2/2), // test
            log(2/1)  // this
        ]
        XCTAssertEqual(idfTermsToNumTextContainingTermExpected, Array(idfTermsToNumTextContainingTerm))
    }
}
