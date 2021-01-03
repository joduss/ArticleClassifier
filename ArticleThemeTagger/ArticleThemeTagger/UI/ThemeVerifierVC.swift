//
//  ThemeVerifierVC.swift
//  ArticleThemeTagger
//
//  Created by Jonathan Duss on 26.01.20.
//  Copyright © 2020 ZaJo. All rights reserved.
//

import Cocoa
import ArticleClassifierCore

class ThemeVerifierVC: NSViewController {
    
    @IBOutlet weak var themeDropdown: NSPopUpButton!
    @IBOutlet weak var verifyStatusLabel: NSTextField!
    @IBOutlet weak var progressLabel: NSTextField!
    @IBOutlet var textView: NSTextView!
    @IBOutlet weak var showAllButton: NSButton!
    
    private let jsonArticlesIO = ArticlesJsonFileIO()

    private var allArticles: [Int : TCVerifiedArticle] = [:]
    private var currentArticle: TCVerifiedArticle?
    private var articlesToVerify: [TCVerifiedArticle] = []

    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    /// Returns a list of articles that hasn't been verified for a given theme, but for which that theme has been predicted.
    private func filterUnverifiedPredictedArticles(with theme:String, articles: [TCVerifiedArticle]) -> [TCVerifiedArticle]{
        return articles.filter({ article in
            article.predictedThemes.contains(theme) && !article.verifiedThemes.contains(theme)
        })
    }
    
    /// Returns a list of articles that hasn't been verified for a given theme
    private func filterUnverifiedArticles(with theme:String, articles: [TCVerifiedArticle]) -> [TCVerifiedArticle]{
        return articles.filter({ article in
            !article.verifiedThemes.contains(theme)
        })
    }
    
    private func displayCurrentArticle() {
        guard let currentArticle = currentArticle else {
            textView.string = "Strange, no article to display!?"
            return
        }
        
        textView.string = currentArticle.title + "\n--------------\n" + currentArticle.summary
        textView.scrollToVisible(NSRect.zero)
    }
    
    @IBAction func showAllButtonClicked(_ sender: NSButton) {
        if sender.state == .on {
            currentArticle = nil
            articlesToVerify = filterUnverifiedArticles(with: themeDropdown.titleOfSelectedItem ?? "", articles: Array(allArticles.values))
            next()
        }
        else {
            currentArticle = nil
            articlesToVerify = filterUnverifiedPredictedArticles(with: themeDropdown.titleOfSelectedItem ?? "", articles: Array(allArticles.values))
            next()
        }
    }
    
    // MARK: - Theme selection
    
    @IBAction func Selected(_ sender: NSPopUpButton) {
        currentArticle = nil
        articlesToVerify = filterUnverifiedPredictedArticles(with: sender.titleOfSelectedItem ?? "", articles: Array(allArticles.values))
        next()
    }
    
    override func keyUp(with event: NSEvent) {
        print(event.keyCode)

        let theme = themeDropdown.titleOfSelectedItem!

        if let specialKey = event.specialKey {
            if specialKey == NSEvent.SpecialKey.rightArrow {
                next()
            }
            return
        }

        guard currentArticle != nil else {
            textView.string = "No article to verify"
            return
        }

        switch event.characters {
        case "y":
            if !self.currentArticle!.verifiedThemes.contains(theme) {
                self.currentArticle!.verifiedThemes.append(theme)
            }
            if !currentArticle!.themes.contains(theme) {
                currentArticle!.themes.append(theme)
            }
            verifyStatusLabel.stringValue = "Approved"

        case "v":
            if !self.currentArticle!.verifiedThemes.contains(theme) {
                self.currentArticle!.verifiedThemes.append(theme)
            }
            self.currentArticle?.themes.removeAll(where: {theme in theme == themeDropdown.titleOfSelectedItem!})
            verifyStatusLabel.stringValue = "Disapproved"
        default:
            break;
        }
    }
    
    //MARK: - Article Loading / Saving
    
    @IBAction func openArticleFile(_ sender: Any) {
        
        allArticles.removeAll()
        currentArticle = nil
        articlesToVerify.removeAll()
        
        let articles = showOpenDialog()
        
        for article in articles {
            allArticles[article.hashValue] = article
        }
        
        let themes = ArticleTheme.allThemes
        
        themeDropdown.addItems(withTitles: themes.map({$0.key}))
        themeDropdown.selectItem(withTitle: themes.first?.key ?? "error!")
        
        articlesToVerify = filterUnverifiedPredictedArticles(with: themes.first?.key ?? "", articles: Array(allArticles.values))
        
        guard articlesToVerify.count > 0 else {
            textView.string = "No articles to verify!"
            return
        }
        
        next()
    }
    
    private func showOpenDialog() -> [TCVerifiedArticle] {
        
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
                    
                    let articles: [TCVerifiedArticle] = try jsonArticlesIO.loadVerifiedArticlesFrom(fileLocation: path)
                    return articles
                }
            }
        }
        catch {
            self.textView.string = "\(error)"
        }
        return []
    }
    
    private func next() {
        
        guard articlesToVerify.count > 0 else {
            textView.string = "No more article to verify"
            progressLabel.stringValue = "\(articlesToVerify.count)"
            return
        }
        
        guard let currentArticle = currentArticle else {
            self.currentArticle = articlesToVerify.first
            displayCurrentArticle()
            progressLabel.stringValue = "\(articlesToVerify.count)"
            return
        }
        
        guard (currentArticle.verifiedThemes.contains(themeDropdown.titleOfSelectedItem ?? "")) else {
            let alert = NSAlert.init()
            alert.messageText = "Please approve or disapprove the theme"
            alert.runModal()
            return
        }
        
        // Replace by the new theme updated!
        allArticles[currentArticle.hashValue] = currentArticle
        verifyStatusLabel.stringValue = "Not verified"
        articlesToVerify.remove(at: 0)
        self.currentArticle = articlesToVerify.first
        progressLabel.stringValue = "\(articlesToVerify.count)"
        
        displayCurrentArticle()
    }
        
    @IBAction func saveArticles(_ sender: Any) {
        let dialog = NSSavePanel();
        
        dialog.title                   = "Choose a .json file";
        dialog.showsResizeIndicator    = true;
        dialog.showsHiddenFiles        = false;
        dialog.canCreateDirectories    = true;
        dialog.allowedFileTypes        = ["json"];
        
        do {
            if (dialog.runModal() == NSApplication.ModalResponse.OK) {
                let result = dialog.url // Pathname of the file
                
                if (result != nil) {
                    let path = result!.path
                    try jsonArticlesIO.WriteToFile(articles: Array(allArticles.values), at: path)
                }
            }
        }
        catch {
            self.textView.string = "\(error)"
        }
        
    }
}
