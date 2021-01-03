//
//  ViewController.swift
//  FetchNewArticleToJson
//
//  Created by Jonathan Duss on 27.08.18.
//  Copyright © 2018 Swizapp. All rights reserved.
//

import Cocoa
import ArticleClassifierCore


class ViewController: NSViewController {
    
    private static let FileExtension = "json"
    
    private let feedSupport = FeedSupport(supportedFeedPlist: "RSSFeeds")

    private var articles: [TCVerifiedArticle] = []
    
    
    private let articlesFetcher = ArticleFetcher()
    private let jsonArticlesIO = ArticlesJsonFileIO()
    let converter = ArticleJsonConverter()
    
    
    //MARK: - outlets
    //===================================================================
    
    @IBOutlet weak var segmentedControl: NSSegmentedControl!
    @IBOutlet var textView: NSTextView!
    @IBOutlet weak var articlesCountTF: NSTextField!
    
    
    //MARK: - Actions
    //===================================================================
    
    @IBAction func save(_ sender: Any) {
        let dialog = NSSavePanel();
                
        dialog.title                   = "Choose a .json file";
        dialog.showsResizeIndicator    = true;
        dialog.showsHiddenFiles        = false;
        dialog.canCreateDirectories    = true;
        dialog.allowedFileTypes        = [ViewController.FileExtension];
        
        do {
            if (dialog.runModal() == NSApplication.ModalResponse.OK) {
                let result = dialog.url // Pathname of the file
                
                if (result != nil) {
                    let path = result!.path
                    try jsonArticlesIO.WriteToFile(articles: articles, at: path)
                }
            }
        }
        catch {
            self.showError(error)
        }
    }
    
    @IBAction func loadExistingFile(_ sender: Any) {
        let dialog = NSOpenPanel();
        
        dialog.title                   = "Choose a .json file";
        dialog.showsResizeIndicator    = true;
        dialog.showsHiddenFiles        = false;
        dialog.canChooseDirectories    = true;
        dialog.canCreateDirectories    = true;
        dialog.allowsMultipleSelection = false;
        dialog.allowedFileTypes        = [ViewController.FileExtension];
        
        do {
            if (dialog.runModal() == NSApplication.ModalResponse.OK) {
                let result = dialog.url // Pathname of the file
                
                if (result != nil) {
                    let path = result!.path
                    self.articles = try jsonArticlesIO.loadVerifiedArticlesFrom(fileLocation: path)
                    
                    let jsonString = String(data: ArticleJsonConverter.convertToJson(articles: self.articles), encoding: .utf8)
                    self.textView.string = jsonString ?? "error"
                    
                }
            }
        }
        catch {
            self.showError(error)
        }
        
        self.articlesCountTF.stringValue = "\(articles.count)"
    }
    
    @IBAction func fetchNewArticles(_ sender: Any) {
        
        let feeds = feedSupport.getFeedPO().filter({$0.language == segmentedControl.label(forSegment: segmentedControl.selectedSegment)})
        
        articlesFetcher.fetchArticles(of: feeds, onProgress: {
            self.textView.string = "Fetching (\($0))"}, completion: { newArticles in
                self.articles = self.mergeArticles(mergeInto: self.articles, from: newArticles)
                let json = String(data: ArticleJsonConverter.convertToJson(articles: self.articles), encoding: .utf8)
                self.textView.string = json ?? "Nothing"
                self.articlesCountTF.stringValue = "\(self.articles.count)"
        })
    }
    
    //MARK: - Helpers
    //===================================================================
    
    private func showError(_ error: Error) {
        if let error = error as? NAError {
            switch error {
            case .error(let message):
                textView.string = message
            }
        }
        else {
            textView.string = error.localizedDescription
        }
    }
    
    // Merge article from "from" into "mergeInto" ensuring that article with same summary are added only once.
    private func mergeArticles(mergeInto: [TCVerifiedArticle], from: [TCVerifiedArticle]) -> [TCVerifiedArticle] {
        
        var mergedArticles = mergeInto
        mergedArticles.reserveCapacity(mergeInto.count + from.count)
        
        
        var mergedArticlesDictionary: [Int : TCVerifiedArticle] = Dictionary<Int, TCVerifiedArticle>(minimumCapacity: mergeInto.count + from.count)
        
        for article in mergedArticles {
            mergedArticlesDictionary[article.summary.hashValue] = article
        }
        
        for articleFrom in from {
            if mergedArticlesDictionary[articleFrom.summary.hashValue] == nil {
                mergedArticles.append(articleFrom)
            }
        }
        return mergedArticles
    }
    
    
    
}

