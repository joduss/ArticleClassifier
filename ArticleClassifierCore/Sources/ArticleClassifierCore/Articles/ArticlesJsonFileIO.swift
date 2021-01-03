//
//  PreviousArticlesJsonLoader.swift
//  FetchNewArticleToJson
//
//  Created by Jonathan Duss on 30.08.18.
//  Copyright © 2018 Swizapp. All rights reserved.
//

import Foundation


public class ArticlesJsonFileIO {
        
    public init() { }

    public func loadArticlesFrom(fileLocation: String) throws -> [TCArticle] {
        let fileManager = FileManager.default
        if let jsonData = fileManager.contents(atPath: fileLocation) {
            let decoder = JSONDecoder()
            return try decoder.decode([TCArticle].self, from: jsonData)
        }
        else {
            throw NAError.error(message: "Error while getting the data at \(fileLocation)")
        }
    }
    
    public func WriteToFile(articles: [TCArticle], at fileLocation: String) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(articles)
        
        let filemanager = FileManager.default
        if filemanager.fileExists(atPath: fileLocation) {
            try data.write(to: URL(fileURLWithPath: fileLocation))
        }
        filemanager.createFile(atPath: fileLocation, contents: data, attributes: nil)
    }
    
    public func loadVerifiedArticlesFrom(fileLocation: String) throws -> [TCVerifiedArticle] {
        let fileManager = FileManager.default
        if let jsonData = fileManager.contents(atPath: fileLocation) {
            return ArticleJsonConverter.verifiedArticleFromJson(jsonData: jsonData)
        }
        else {
            throw NAError.error(message: "Error while getting the data at \(fileLocation)")
        }
    }
    
    public func WriteToFile(articles: [TCVerifiedArticle], at fileLocation: String) throws {        
        let data: Data = ArticleJsonConverter.convertToJson(articles: articles)
        let filemanager = FileManager.default
        
        if filemanager.fileExists(atPath: fileLocation) {
            try data.write(to: URL(fileURLWithPath: fileLocation))
        }
        
        filemanager.createFile(atPath: fileLocation, contents: data, attributes: nil)
    }
}
