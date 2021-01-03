//
//  ArticlePreprocessorTool.swift
//  ArticlePreprocessorTool
//
//  Created by Jonathan Duss on 23.05.20.
//  Copyright © 2020 ZaJo. All rights reserved.
//

import Foundation
import ArticleClassifierCore

public class ArticlePreprocessorTool {
    
    // Concurrency management
    private let group = DispatchGroup()
    private let semaphore = DispatchSemaphore(value: 1)
    private let arraySemaphore = DispatchSemaphore(value: 1)
    private let queue = OperationQueue()
    
    private let articlesToProcess: [TCVerifiedArticle]
    private let firstArticle: TCVerifiedArticle
    private let processor: ACTextPreprocessor
    
    private var processedArticle: [TCVerifiedArticle]
    private let outputPath: String

    
    public init(inputFilePath: String, outputFilePath: String) throws {
        queue.maxConcurrentOperationCount = 12
        
        self.outputPath = outputFilePath
        
        articlesToProcess = try ArticlesJsonFileIO().loadVerifiedArticlesFrom(fileLocation: inputFilePath)
        firstArticle = articlesToProcess.first!
        
        processor = ACTextPreprocessor(representativeText: firstArticle.title + firstArticle.summary)
        processor.lemmatizer = ACLemmatizerV2(language: processor.language)
        
        processedArticle = [TCVerifiedArticle](articlesToProcess)
    }
    
    
    /// Start processing the file given in the constructor.
    public func process() throws {
        queue.progress.totalUnitCount = Int64(articlesToProcess.count)

        for i in 0..<articlesToProcess.endIndex {
            let index = i
            group.enter()
            queue.addOperation {
                let article = self.articlesToProcess[index]
                self.setArticle(article: self.processArticle(article),
                                arrayIdx: index)

                if (self.queue.progress.completedUnitCount % 10 == 0) {
                    let message = "Finished processing article \(self.queue.progress.completedUnitCount)/\(self.queue.progress.totalUnitCount) (\(self.queue.progress.fractionCompleted * 100))"
                    print(message)
                    setbuf(stdout, nil);
                }
                self.group.leave()
            }
        }

        print("Now just waiting that all are processed.")
        group.wait()
        print("Continue... all articles have been now processed.")

        try ArticlesJsonFileIO().WriteToFile(articles: processedArticle, at: self.outputPath)
    }
    
    private func setArticle(article: TCVerifiedArticle, arrayIdx: Int) {
        arraySemaphore.wait()
        self.processedArticle[arrayIdx] = article
        arraySemaphore.signal()
    }

    
    private func processArticle(_ article: TCVerifiedArticle) -> TCVerifiedArticle {
        let title = processor.process(text: article.title)
        let summary = processor.process(text: article.summary)
        return article.articleWithUpdated(title: title, summary: summary)
    }


    private func getUpdateProgress(){
        semaphore.wait()
        queue.progress.completedUnitCount += 1
        semaphore.signal()
    }
}
