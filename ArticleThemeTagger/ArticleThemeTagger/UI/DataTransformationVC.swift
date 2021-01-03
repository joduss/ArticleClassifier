//
//  DataTransformationVC.swift
//  ArticleThemeTagger
//
//  Created by Jonathan Duss on 26.01.20.
//  Copyright © 2020 ZaJo. All rights reserved.
//

import Cocoa
import ArticleClassifierCore

class DataTransformationVC: NSViewController {

    private let jsonArticlesIO = ArticlesJsonFileIO()
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
    }
    
    @IBAction func transformArticlesToVerifiedArticles(_ sender: Any) {
        
        let dialog = NSOpenPanel();
        
        dialog.title                   = "Choose a .json file";
        dialog.showsResizeIndicator    = true;
        dialog.showsHiddenFiles        = false;
        dialog.canChooseDirectories    = true;
        dialog.canCreateDirectories    = true;
        dialog.allowsMultipleSelection = false;
        dialog.allowedFileTypes        = ["json"];
        
        do {
            if (dialog.runModal() == NSApplication.ModalResponse.OK) {
                let result = dialog.url // Pathname of the file
                
                if (result != nil) {
                    let path = result!.path
                    let verifiedArticleOutputPath = path.replacingOccurrences(of: ".json", with: "") + "-verified.json"

                    let articles: [TCArticle] = try jsonArticlesIO.loadArticlesFrom(fileLocation: path)
                    var verifiedArticles: [TCVerifiedArticle] = []
                    
                    
                    for article in articles {
                        var verifiedArticle = TCVerifiedArticle(title: article.title, summary: article.summary, themes: article.themes)
                        if (article.themes.count > 0) {
                            verifiedArticle.verifiedThemes.append(contentsOf: ArticleTheme.allThemes.map({$0.key}))
                        }
                        verifiedArticles.append(verifiedArticle)
                    }
                    
                    try jsonArticlesIO.WriteToFile(articles: verifiedArticles, at: verifiedArticleOutputPath)
                }
            }
        }
        catch {
            self.showError(error)
        }
    }
    
    //MARK: - Helpers
    //===================================================================
    
    private func showError(_ error: Error) {
        NSAlert(error: error).runModal()
    }
}
