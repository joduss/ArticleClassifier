//
//  articlePreprocessorToolTests.swift
//  articlePreprocessorToolTests
//
//  Created by Jonathan Duss on 23.05.20.
//  Copyright © 2020 ZaJo. All rights reserved.
//

import XCTest
import ArticleClassifierCore

class articlePreprocessorToolTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    func testManual() throws {
        let text = "L'iphone qu'Apple a présenté hier ne sera disponible qu'au moins de septembre 2020."
        
        let preprocessor = ACTextPreprocessor(representativeText: text)
        let result = preprocessor.process(text: text)
        
        print(result)
        print("OK")
    }

    func testProcess() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        
        let article1 = TCVerifiedArticle(title: "Les titres pour article 1", summary: "Les résumés pour article 1", themes: [ArticleTheme.amazon.key])
        let article2 = TCVerifiedArticle(title: "Les titres pour article 2", summary: "Les résumés pour article 2", themes: [ArticleTheme.amazon.key])
        
        let article1Id = article1.id
        let article2Id = article2.id
        
        let inputFilePath = createTempFilePath()
        let outputFilePath = createTempFilePath()

        try ArticlesJsonFileIO().WriteToFile(articles: [article1, article2], at: inputFilePath)
        
        guard let processor = try? ArticlePreprocessorTool(inputFilePath: inputFilePath, outputFilePath: outputFilePath) else {
            assertionFailure("Should not be any exception.")
            return
        }
        
        try processor.process()
        
        let processedArticles = try ArticlesJsonFileIO().loadVerifiedArticlesFrom(fileLocation: outputFilePath)
        
        XCTAssertEqual(2, processedArticles.count)
        let processedId = processedArticles.map({$0.id})
        XCTAssertTrue(processedId.contains(article1.id))
        XCTAssertTrue(processedId.contains(article2.id))
        
        XCTAssertEqual("résumé article 1", processedArticles.filter({$0.id == article1Id}).first?.summary)
        XCTAssertEqual("résumé article 2", processedArticles.filter({$0.id == article2Id}).first?.summary)
        
        XCTAssertEqual("titre article 1", processedArticles.filter({$0.id == article1Id}).first?.title ?? "")
        XCTAssertEqual("titre article 2", processedArticles.filter({$0.id == article2Id}).first?.title ?? "")
    }
    
    /// Just creates a path to a new temporary file. The file is not created; only the path is returned.
    private func createTempFilePath() -> String {
        let fileName = ProcessInfo.processInfo.globallyUniqueString
        return URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true).appendingPathComponent(fileName).path
    }
}
