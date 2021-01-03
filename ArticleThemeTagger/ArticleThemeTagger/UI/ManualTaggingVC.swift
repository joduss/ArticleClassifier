//
//  ViewController.swift
//  ArticleThemeTruthMaker
//
//  Created by Jonathan Duss on 22.06.19.
//  Copyright © 2019 ZaJo. All rights reserved.
//

import Cocoa
import ArticleClassifierCore

fileprivate enum NavigationMode {
    case InOrder
    case Random
    case RandomNoTheme
}

class ManualTaggingVC: NSViewController {

    @IBOutlet weak var filterTextView: NSTextField!
    @IBOutlet weak var buttonsView: NSView!
    @IBOutlet var textView: NSTextView!
    @IBOutlet weak var gridview: NSGridView!
    @IBOutlet weak var dropdown: NSPopUpButton!
    @IBOutlet weak var statsLabel: NSTextField!
    @IBOutlet weak var goToTF: NSTextField!
    
    private let DropDownOptionInOrder = "In Order"
    private let DropDownOptionRandom = "Random"
    private let DropDownOptionRandomNoTheme = "Random no theme"
    
    private let rowCount = 16.0
    
    private let allThemes = ArticleTheme.allThemes

    private let jsonArticlesIO = ArticlesJsonFileIO()
    private var articleFilePath: String? = nil
    
    private var articleList = ArticleList()

    private var filteredArticleList = ArticleList()
    
    private var countArticleWithoutThemes = 0
    
    fileprivate var editingText = false

    
    private var currentArticle: TCVerifiedArticle? {
        didSet {
            guard let currentArticle = self.currentArticle else {
                textView.string = "No Articles"
                updateStats()
                updateCheckboxes()
                return
            }
            
            var summaryCleaned = currentArticle.summary
            summaryCleaned = summaryCleaned.replacingOccurrences(of: "\n", with: "\\n")
            summaryCleaned = summaryCleaned.replacingOccurrences(of: "\t", with: "")

            textView.string = """
            \(currentArticle.title)
            
            \(summaryCleaned)
            """
            goToTF.stringValue = filteredArticleList.index(of: currentArticle).description
            updateCheckboxes()
            
            updateStats()
        }
    }
    
    private var navigationMode: NavigationMode {
        switch dropdown.titleOfSelectedItem {
        case DropDownOptionInOrder:
            return .InOrder
        case DropDownOptionRandom:
            return .Random
        case DropDownOptionRandomNoTheme:
            return .RandomNoTheme
        default:
            return .InOrder
        }
    }
    
    // MARK: - View Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
                
        dropdown.removeAllItems()
        dropdown.addItem(withTitle: DropDownOptionInOrder)
        dropdown.addItem(withTitle: DropDownOptionRandom)
        dropdown.addItem(withTitle: DropDownOptionRandomNoTheme)
        dropdown.selectItem(withTag: 0)
        
        filterTextView.delegate = self
        
        updateCheckboxes()
        updateStats()
    }
    
    // MARK: - Article Filtering
    
    private func filterArticles() {
        
        // Empty the dic and hash arrayt
        var filteredThemes: [String] = []
        
        if filterTextView.stringValue != "" {
            filteredThemes.append(contentsOf: filterTextView.stringValue.components(separatedBy: ", "))
        }
        
        filteredArticleList = articleList.filteredByMissingThemes(filteredThemes)
        
        guard filteredArticleList.isEmpty else {
            currentArticle = nil
            return
        }
        
        currentArticle = filteredArticleList.first
    }
    
    // MARK: - UI Update
    //===================================================================

    private func updateCheckboxes() {
        var currentPosition = 0.0

        for theme in allThemes {
            var row = 0.0
            var col = 0.0
            
            if currentPosition != 0 {
                col = floor(currentPosition / rowCount)
                row = currentPosition - col * rowCount
            }
            
            let view = gridview.cell(atColumnIndex: Int(col), rowIndex: Int(row))
            
            var button: NSButton! = view.contentView as? NSButton
            
            if button == nil {
                button = NSButton(checkboxWithTitle: theme.key, target: self, action: #selector(themeSelected))
                view.contentView = button
                button.translatesAutoresizingMaskIntoConstraints = false
                button.widthAnchor.constraint(equalToConstant: 150).isActive = true
            }
            
            
            if let article = currentArticle, article.themes.contains(theme.key) {
                button.state = .on
            }
            else {
                button.state = .off
            }
            
            currentPosition += 1
        }
    }
    
    /// Updates stats
    private func updateStats() {
        
        if filteredArticleList.isEmpty {
            statsLabel.stringValue = "there are no articles"
        }
        
        countArticleWithoutThemes = 0
        for article in filteredArticleList.articles {
            if article.themes.isEmpty {
                countArticleWithoutThemes += 1
            }
        }
        
        statsLabel.stringValue = "Without theme: \(countArticleWithoutThemes) / \(filteredArticleList.count)"
    }

    
    // MARK: - Load/Save articles.
    //===================================================================
    
    /// Load all the articles
    @IBAction func load(_ sender: Any) {
        
        // Cleanup
        
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
                    self.articleFilePath = path

                    let articles = try jsonArticlesIO.loadVerifiedArticlesFrom(fileLocation: path)
                    articleList = ArticleList(articles: articles)
                    filterArticles()
                }
            }
        }
        catch {
            self.showError(error)
        }
        
        self.updateStats()
    }
    
    @IBAction func save(_ sender: Any) {
        guard let path = self.articleFilePath else {
            return
        }
        do {
            // We want to keep the same order in the json.
            try jsonArticlesIO.WriteToFile(articles: articleList.orderedArticles, at: path)
        }
        catch {
            statsLabel.stringValue = "Saving failed. \(error.localizedDescription)"
        }
    }
    
    // MARK: - Article navigation.
    
    @IBAction func next(_ sender: Any) {
        guard filteredArticleList.isEmpty == false else {
            textView.string = "no articles. Please load something first!"
            return
        }
        
        switch navigationMode {
        case .InOrder:
            guard let currentArticle = self.currentArticle else { return }
            self.currentArticle = filteredArticleList.next(after: currentArticle)
        case .Random:
            self.currentArticle = filteredArticleList.randomArticle
        case .RandomNoTheme:
            self.currentArticle = filteredArticleList.randomArticleWithoutTheme
        }
        
        updateCheckboxes()
    }
    
    private func previous() {
        
        guard filteredArticleList.isEmpty == false else {
            textView.string = "no articles. Please load something first!"
            return
        }
        
        guard let currentArticle = self.currentArticle else { return }
        
        self.currentArticle = filteredArticleList.previous(before: currentArticle)
    }

    @IBAction func GoTo(_ sender: Any) {
        let idx = Int(goToTF.stringValue) ?? 0
        
        guard idx < filteredArticleList.count else {
            textView.string = "Index out of bound!"
            return
        }
        
        currentArticle = filteredArticleList.get(at: idx)
    }
    
    // MARK: - Theme selection
    //===================================================================

    override func keyUp(with event: NSEvent) {
        
        if (editingText) { return }
        
        print(event.keyCode)

        if let specialKey = event.specialKey {
            if specialKey == NSEvent.SpecialKey.leftArrow {
                previous()
            }
            else if specialKey == NSEvent.SpecialKey.rightArrow {
                next(self)
            }

            return
        }

        guard let character = event.characters?.first else {
            return
        }

        var themeFromKey: ArticleTheme?

        switch event.characters {
        case "a":
            themeFromKey = ArticleTheme.apple
        case "t":
            themeFromKey = ArticleTheme.tablet
        case "f":
            themeFromKey = ArticleTheme.facebook
        case "m":
            themeFromKey = ArticleTheme.mac
        case "g":
            themeFromKey = ArticleTheme.google
        case "i":
            themeFromKey = ArticleTheme.iPhone
        case "r":
            themeFromKey = ArticleTheme.rumor
        case "k":
            themeFromKey = ArticleTheme.keynote
        case "p":
            themeFromKey = ArticleTheme.iPad
        case "e":
            themeFromKey = ArticleTheme.economyPolitic
        case "c":
            themeFromKey = ArticleTheme.computer
        case "w":
            themeFromKey = ArticleTheme.watch
        case "n":
            themeFromKey = ArticleTheme.netflix
        case "b":
            themeFromKey = ArticleTheme.beta
        case "d":
            themeFromKey = ArticleTheme.disneyPlus
        case "z":
            themeFromKey = ArticleTheme.ios
        case "u":
            themeFromKey = ArticleTheme.macos
        case "s":
            themeFromKey = ArticleTheme.smartphone
        case "h":
            themeFromKey = ArticleTheme.health
        case "l":
            themeFromKey = ArticleTheme.lawsuitLegal
        default:
            break
        }

        guard let theme = themeFromKey else {
            return
        }

        if let currentArticle = self.currentArticle {
            if let themeIdx = currentArticle.themes.firstIndex(of: theme.key) {
                currentArticle.themes.remove(at: themeIdx)
            }
            else {
                currentArticle.themes.append(theme.key)
            }
        }

        currentArticle?.verifiedThemes = ArticleTheme.allThemes.map({$0.key})

        updateCheckboxes()
    }

    @objc
    private func themeSelected(sender: NSButton) {
        guard let article = currentArticle else {
            return
        }
        
        let themes = article.themes
        let clickedTheme = ArticleTheme.init(key: sender.title)
        
        if sender.state == .on {
            currentArticle?.themes.append(clickedTheme.key)
        }
        else {
            currentArticle?.themes.remove(at: themes.firstIndex(of: clickedTheme.key)!)
        }
        
        currentArticle?.verifiedThemes = ArticleTheme.allThemes.map({$0.key})
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

}

extension ManualTaggingVC: NSTextFieldDelegate {
    
    func controlTextDidBeginEditing(_ obj: Notification) {
        editingText = true
    }
    
    func controlTextDidEndEditing(_ obj: Notification) {
        editingText = false
        filterArticles()
    }
}

